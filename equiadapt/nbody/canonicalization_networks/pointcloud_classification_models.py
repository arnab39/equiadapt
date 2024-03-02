

import pytorch_lightning as pl
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from canonical_network.utils import *
from canonical_network.models.pointcloud_networks import VNSmall, PointNetEncoder
from canonical_network.models.vn_layers import *

class BasePointcloudClassificationModel(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.num_classes = hyperparams.num_classes
        self.train_rotation = hyperparams.train_rotation
        self.valid_rotation = hyperparams.valid_rotation
        self.num_points = hyperparams.num_points
        self.learning_rate = hyperparams.learning_rate
        self.hyperparams = hyperparams

    def configure_optimizers(self):
        if self.hyperparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                         weight_decay=self.hyperparams.decay_rate)
            print("Using Adam optimizer")
            return optimizer
        elif self.hyperparams.optimizer == "SGD":
            self.learning_rate *= 100
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9,
                                        weight_decay=self.hyperparams.decay_rate)
            if self.hyperparams.decay_type == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=self.hyperparams.num_epochs, eta_min=1e-3)
            elif self.hyperparams.decay_type == "step":
                scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
            else:
                raise NotImplementedError(f"Unknown learning rate decay {self.hyperparams.decay_type}")
            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "epoch",
            }
            print(f"Using SGD optimizer with {self.hyperparams.decay_type} learning rate scheduler")
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx):
        points, targets = batch
        points, targets = points.float(), targets.long()

        # Augmentations
        trot = None
        if self.train_rotation == "z":
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True, device=self.device)
        elif self.train_rotation == "so3":
            trot = Rotate(R=random_rotations(points.shape[0]), device=self.device)
        if trot is not None:
            points = trot.transform_points(points)
        if self.hyperparams.augment_train_data:
            points = random_point_dropout(points)
            points = random_scale_point_cloud(points)
            points = random_shift_point_cloud(points)
        points = points.transpose(2, 1)
        targets = targets[:, 0]

        # Forward pass
        outputs = self(points)

        # Loss
        loss = self.get_loss(outputs, targets)

        metrics = {"train/loss": loss}
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        self.mean_correct = []
        self.class_acc = np.zeros((self.num_classes,3))

    def validation_step(self, batch, batch_idx):
        points, targets = batch
        points, targets = points.float(), targets.long()

        trot = None
        if self.valid_rotation == "z":
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True, device=self.device)
        elif self.valid_rotation == "so3":
            trot = Rotate(R=random_rotations(points.shape[0]), device=self.device)
        if trot is not None:
            points = trot.transform_points(points)

        points = points.transpose(2, 1)
        targets = targets[:, 0]

        outputs = self(points)
        predictions = self.get_predictions(outputs)

        pred_choice = predictions.data.max(1)[1]
        for cat in np.unique(targets.cpu()):
            classacc = pred_choice[targets == cat].eq(targets[targets == cat].long().data).cpu().sum()
            self.class_acc[cat, 0] += classacc.item() / float(points[targets == cat].size()[0])
            self.class_acc[cat, 1] += 1
        correct = pred_choice.eq(targets.long().data).cpu().sum()
        self.mean_correct.append(correct.item() / float(points.size()[0]))

        return outputs

    def validation_epoch_end(self, outputs):

        self.class_acc[:, 2] = self.class_acc[:, 0] / self.class_acc[:, 1]
        class_acc = np.mean(self.class_acc[:, 2])
        instance_acc = np.mean(self.mean_correct)
        self.log_dict(
            {"valid/instance_accuracy": instance_acc,
             "valid/class_accuracy": class_acc},
            prog_bar=True)

    def get_loss(self, outputs, targets, smoothing=True):
        predictions = self.get_predictions(outputs)
        targets = targets.contiguous().view(-1)
        if smoothing:
            eps = 0.2
            one_hot = torch.zeros_like(predictions).scatter(1, targets.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (self.num_classes - 1)
            log_prb = F.log_softmax(predictions, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(predictions, targets, reduction='mean')

        return loss

    def get_predictions(self, outputs):
        if type(outputs) == list:
            outputs = list(zip(*outputs))
            return torch.cat(outputs[0], dim=0)
        return outputs[0]



class Pointnet(BasePointcloudClassificationModel):
    def __init__(self, hyperparams):
        super(Pointnet, self).__init__(hyperparams)
        self.normal_channel = hyperparams.normal_channel
        self.regularization_transform = hyperparams.regularization_transform

        if self.normal_channel:
            channel = 6
        else:
            channel = 3

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, hyperparams.num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, point_cloud):
        x, trans, trans_feat = self.feat(point_cloud)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

    def get_loss(self, outputs, targets):
        predictions = self.get_predictions(outputs)
        transformation_matrix = outputs[1]

        transformation_loss = 0.001 * self.feature_transform_regularizer(transformation_matrix)
        total_loss = F.nll_loss(predictions, targets) + transformation_loss

        return total_loss

    def feature_transform_regularizer(self, trans):
        d = trans.shape[1]
        I = torch.eye(d)[None, :, :].to(trans.device)
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
        return loss


class DGCNN(BasePointcloudClassificationModel):
    def __init__(self, hyperparams):
        super(DGCNN, self).__init__(hyperparams)
        self.n_knn = hyperparams.n_knn

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, hyperparams.num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.n_knn)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        trans_feat = None

        return x, trans_feat



class PointcloudCanonFunction(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.model_type = hyperparams.canon_model_type
        self.model = {"vn_net": lambda: VNSmall(hyperparams)}[self.model_type]()

    def forward(self, points):
        vectors = self.model(points)
        rotation_vectors = vectors[:, :3]
        translation_vectors = vectors[:, 3:]

        rotation_matrix = self.gram_schmidt(rotation_vectors)
        return rotation_matrix, translation_vectors

    def gram_schmidt(self, vectors):
        v1 = vectors[:, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = (vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1)
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        v3 = (vectors[:, 2] - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1 - torch.sum(vectors[:, 2] * v2,
                                                                                                  dim=1,
                                                                                                  keepdim=True) * v2)
        v3 = v3 / torch.norm(v3, dim=1, keepdim=True)
        return torch.stack([v1, v2, v3], dim=1)


class PointcloudPredFunction(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.model_type = hyperparams.pred_model_type
        self.model = {"pointnet": lambda: Pointnet(hyperparams),
                      "DGCNN": lambda: DGCNN(hyperparams)}[self.model_type]()

    def forward(self, points):
        return self.model(points)


class EquivariantPointcloudModel(BasePointcloudClassificationModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.model = hyperparams.model
        self.hyperparams = hyperparams

        self.canon_function = PointcloudCanonFunction(hyperparams)
        self.pred_function = PointcloudPredFunction(hyperparams)

    def forward(self, point_cloud):
        rotation_matrix, translation_vectors = self.canon_function(point_cloud)
        rotation_matrix_inverse = rotation_matrix.transpose(1, 2)

        # not applying translations
        canonical_point_cloud = torch.bmm(point_cloud.transpose(1, 2), rotation_matrix_inverse)
        canonical_point_cloud = canonical_point_cloud.transpose(1, 2)

        predictions, _ = self.pred_function(canonical_point_cloud)

        return predictions, rotation_matrix


class VNPointnet(BasePointcloudClassificationModel):
    def __init__(self, hyperparams):
        super(VNPointnet, self).__init__(hyperparams)
        self.hyperparams = hyperparams
        if hyperparams.normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(hyperparams, global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024 // 3 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, hyperparams.num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

    def get_loss(self, outputs, targets):
        predictions = self.get_predictions(outputs)
        return F.nll_loss(predictions, targets)







