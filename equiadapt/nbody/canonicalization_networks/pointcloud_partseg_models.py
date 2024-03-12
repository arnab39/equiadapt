
import pytorch_lightning as pl
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from canonical_network.utils import *
from canonical_network.models.pointcloud_networks import STNkd, STN3d, VNSTNkd, Transform_Net, VNSmall
from canonical_network.models.vn_layers import *

SEGMENTATION_CLASSES = {
    "Earphone": [16, 17, 18],
    "Motorbike": [30, 31, 32, 33, 34, 35],
    "Rocket": [41, 42, 43],
    "Car": [8, 9, 10, 11],
    "Laptop": [28, 29],
    "Cap": [6, 7],
    "Skateboard": [44, 45, 46],
    "Mug": [36, 37],
    "Guitar": [19, 20, 21],
    "Bag": [4, 5],
    "Lamp": [24, 25, 26, 27],
    "Table": [47, 48, 49],
    "Airplane": [0, 1, 2, 3],
    "Pistol": [38, 39, 40],
    "Chair": [12, 13, 14, 15],
    "Knife": [22, 23],
}

SEGMENTATION_LABEL_TO_PART = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in SEGMENTATION_CLASSES.keys():
    for label in SEGMENTATION_CLASSES[cat]:
        SEGMENTATION_LABEL_TO_PART[label] = cat

LEARNING_RATE_CLIP = 1e-5
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECCAY = 0.5


class BasePointcloudModel(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.num_parts = hyperparams.num_parts
        self.num_classes = hyperparams.num_classes
        self.train_rotation = hyperparams.train_rotation
        self.valid_rotation = hyperparams.valid_rotation
        self.num_points = hyperparams.num_points
        self.learning_rate = hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None
        self.hyperparams = hyperparams

    def get_predictions(self, outputs):
        if type(outputs) == list:
            outputs = list(zip(*outputs))
            return torch.cat(outputs[0], dim=0)
        return outputs[0]

    def configure_optimizers(self):
        if self.hyperparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.hyperparams.decay_rate)
            print("Using Adam optimizer")
            return optimizer
        elif self.hyperparams.optimizer == "SGD":
            self.learning_rate *= 100
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.hyperparams.decay_rate)
            print("Using SGD optimizer with custom learning rate scheduler")
            return optimizer
        elif self.hyperparams.optimizer == "SGD_built_in":
            self.learning_rate *= 100
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.hyperparams.decay_rate)
            if self.hyperparams.decay_type == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=self.hyperparams.num_epochs, eta_min=1e-3)
            elif self.hyperparams.decay_type == "step":
                scheduler = MultiStepLR(optimizer,
                                        milestones=[self.trainer.max_epochs // 6, self.trainer.max_epochs // 3,
                                                    self.trainer.max_epochs // 2], gamma=0.1)
            else:
                raise NotImplementedError(f"Unknown learning rate decay {self.hyperparams.decay_type}")
            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "epoch",
            }
            print(f"Using SGD optimizer with {self.hyperparams.lr_decay} learning rate scheduler")
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        else:
            raise NotImplementedError

    def on_train_epoch_start(self):
        if self.hyperparams.optimizer == "SGD":
            lr = max(self.learning_rate * (self.hyperparams.lr_decay ** (self.current_epoch // self.hyperparams.step_size)), LEARNING_RATE_CLIP)
            for param_group in self.optimizers().param_groups:
                param_group['lr'] = lr
            momentum = max(MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (self.current_epoch // self.hyperparams.step_size)), 0.01)
            self.apply(lambda x: bn_momentum_adjust(x, momentum))
        lr = self.optimizers().param_groups[0]['lr']
        print(f"Learning rate in epoch {self.current_epoch} is {lr}")

    def training_step(self, batch, batch_idx):
        points, label, targets = batch
        points, label, targets = points.float(), label.long(), targets.long()

        # Augmentations
        trot = None
        if self.train_rotation == "z":
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True, device=self.device)
        elif self.train_rotation == "so3":
            trot = Rotate(R=random_rotations(points.shape[0]), device=self.device)
        if trot is not None:
            points = trot.transform_points(points)
        if self.hyperparams.augment_train_data:
            points = random_scale_point_cloud(points)
            points = random_shift_point_cloud(points)
        points = points.transpose(2, 1)

        # Forward pass
        ont_hot_labels = to_categorical(label, self.num_classes).type_as(label)
        outputs = self(points, ont_hot_labels)

        # Loss
        loss = self.get_loss(outputs, targets)

        metrics = {"train/loss": loss}
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        self.total_correct = 0
        self.total_seen = 0
        self.total_seen_class = [0 for _ in range(self.num_parts)]
        self.total_correct_class = [0 for _ in range(self.num_parts)]
        self.shape_ious = {cat: [] for cat in SEGMENTATION_CLASSES.keys()}

    def validation_step(self, batch, batch_idx):
        points, label, targets = batch
        points, label, targets = points.float(), label.long(), targets.long()

        trot = None
        if self.valid_rotation == "z":
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True, device=self.device)
        elif self.valid_rotation == "so3":
            trot = Rotate(R=random_rotations(points.shape[0]), device=self.device)
        if trot is not None:
            points = trot.transform_points(points)

        points = points.transpose(2, 1)
        ont_hot_labels = to_categorical(label, self.num_classes).type_as(label)
        outputs = self(points, ont_hot_labels)
        predictions = self.get_predictions(outputs)

        loss = self.get_loss(outputs, targets)
        self.append_to_metric_lists(points, predictions, targets)

        metrics = {"valid/loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return outputs

    def validation_epoch_end(self, outputs):

        accuracy, class_avg_accuracy, class_avg_iou, instance_avg_iou = self.get_metrics()
        self.log_dict(
            {"valid/accuracy": accuracy, "valid/class_avg_iou": class_avg_iou, "valid/instance_avg_iou": instance_avg_iou},
            prog_bar=True)

    def get_loss(self, outputs, targets, smoothing=True):
        ''' Calculate cross entropy loss and apply label smoothing. '''
        predictions = self.get_predictions(outputs)
        predictions = predictions.flatten(0, 1)
        targets = targets.flatten(0, 1)

        if smoothing:
            eps = 0.2

            one_hot = torch.zeros_like(predictions).scatter(1, targets.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (self.num_classes - 1)
            log_prb = F.log_softmax(predictions, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(predictions, targets, reduction='mean')

        return loss

    def append_to_metric_lists(self, points, predictions, targets):
        cur_batch_size, _, NUM_POINT = points.shape

        cur_pred_val = predictions.cpu().data.numpy()
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        targets = targets.cpu().data.numpy()
        for i in range(cur_batch_size):
            cat = SEGMENTATION_LABEL_TO_PART[targets[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, SEGMENTATION_CLASSES[cat]], 1) + SEGMENTATION_CLASSES[cat][0]

        correct = np.sum(cur_pred_val == targets)
        self.total_correct += correct
        self.total_seen += (cur_batch_size * NUM_POINT)

        for l in range(self.num_parts):
            self.total_seen_class[l] += np.sum(targets == l)
            self.total_correct_class[l] += (np.sum((cur_pred_val == l) & (targets == l)))

        for i in range(cur_batch_size):
            segp = cur_pred_val[i, :]
            segl = targets[i, :]
            cat = SEGMENTATION_LABEL_TO_PART[segl[0]]
            part_ious = [0.0 for _ in range(len(SEGMENTATION_CLASSES[cat]))]
            for l in SEGMENTATION_CLASSES[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - SEGMENTATION_CLASSES[cat][0]] = 1.0
                else:
                    part_ious[l - SEGMENTATION_CLASSES[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l))
                    )
            self.shape_ious[cat].append(np.mean(part_ious))

    def get_metrics(self):
        all_shape_ious = []
        for cat in self.shape_ious.keys():
            for iou in self.shape_ious[cat]:
                all_shape_ious.append(iou)
            self.shape_ious[cat] = np.mean(self.shape_ious[cat])
        mean_shape_ious = np.mean(list(self.shape_ious.values()))
        accuracy = self.total_correct / float(self.total_seen)
        class_avg_accuracies = np.mean(np.array(self.total_correct_class) / np.array(self.total_seen_class, dtype=np.float))
        class_avg_ious = mean_shape_ious
        instance_avg_iou = np.mean(all_shape_ious)
        return accuracy, class_avg_accuracies, class_avg_ious, instance_avg_iou


class Pointnet(BasePointcloudModel):
    def __init__(self, hyperparams):
        super(Pointnet, self).__init__(hyperparams)
        self.model = "pointnet"
        self.num_parts = hyperparams.num_parts
        self.normal_channel = hyperparams.normal_channel
        self.regularization_transform = hyperparams.regularization_transform

        if self.normal_channel:
            channel = 6
        else:
            channel = 3

        self.stn = STN3d(channel) if self.regularization_transform else None
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128) if self.regularization_transform else None
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, self.num_parts, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()
        if self.regularization_transform:
            trans = self.stn(point_cloud)
            point_cloud = point_cloud.transpose(2, 1)
            if D > 3:
                point_cloud, feature = point_cloud.split(3, dim=2)
            point_cloud = torch.bmm(point_cloud, trans)
            if D > 3:
                point_cloud = torch.cat([point_cloud, feature], dim=2)
            point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        if self.regularization_transform:
            trans_feat = self.fstn(out3)
            x = out3.transpose(2, 1)
            net_transformed = torch.bmm(x, trans_feat)
            net_transformed = net_transformed.transpose(2, 1)
        else:
            net_transformed = out3
            trans_feat = None

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        out_max = torch.cat([out_max, label.squeeze(1)], 1)
        expand = out_max.view(-1, 2048 + 16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.num_parts), dim=-1)
        net = net.view(B, N, self.num_parts)  # [B, N, 50]

        return net, trans_feat

    def get_loss(self, outputs, targets):
        predictions = outputs[0].permute(0, 2, 1)
        transformation_matrix = outputs[1]

        transformation_loss = (
            self.regularization_transform * self.feature_transform_regularizer(transformation_matrix)
            if self.regularization_transform
            else 0
        )
        total_loss = F.nll_loss(predictions, targets) + transformation_loss

        return total_loss

    def feature_transform_regularizer(self, trans):
        d = trans.shape[1]
        I = torch.eye(d)[None, :, :].to(trans.device)
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
        return loss

class DGCNN(BasePointcloudModel):
    def __init__(self, hyperparams):
        super(DGCNN, self).__init__(hyperparams)
        self.model = "DGCNN"
        self.num_parts = hyperparams.num_parts
        self.normal_channel = hyperparams.normal_channel
        self.regularization_transform = hyperparams.regularization_transform
        self.n_knn = hyperparams.n_knn
        self.transform_net = Transform_Net(hyperparams)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,

                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.num_parts, kernel_size=1, bias=False)

    def forward(self, x, l):

        batch_size, _, num_points = x.shape

        x0 = get_graph_feature(x, k=self.n_knn)
        t = self.transform_net(x0)
        x = x.transpose(2, 1)
        x = torch.bmm(x, t)
        x = x.transpose(2, 1)

        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x)
        x = x.max(dim=-1, keepdim=True)[0]

        l = l.view(batch_size, -1, 1).type_as(x)
        l = self.conv7(l)

        x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        x = x.transpose(1, 2)

        trans_feat = None
        return x, trans_feat



class PointcloudCanonFunction(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.model_type = hyperparams.canon_model_type
        self.model = {"vn_pointnet": lambda: VNSmall(hyperparams)}[self.model_type]()

    def forward(self, points, labels):
        vectors = self.model(points, labels)
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

    def forward(self, points, labels):
        return self.model(points, labels)


class EquivariantPointcloudModel(BasePointcloudModel):
    def __init__(self, hyperparams):
        super(EquivariantPointcloudModel, self).__init__(hyperparams)
        self.model = hyperparams.model
        self.hyperparams = hyperparams
        self.num_parts = hyperparams.num_parts

        self.canon_function = PointcloudCanonFunction(hyperparams)
        self.pred_function = PointcloudPredFunction(hyperparams)

    def forward(self, point_cloud, label):
        rotation_matrix, translation_vectors = self.canon_function(point_cloud, label)
        rotation_matrix_inverse = rotation_matrix.transpose(1, 2)

        # not applying translations
        canonical_point_cloud = torch.bmm(point_cloud.transpose(1, 2), rotation_matrix_inverse)
        canonical_point_cloud = canonical_point_cloud.transpose(1, 2)

        return self.pred_function(canonical_point_cloud, label)[0], rotation_matrix



class VNPointnet(BasePointcloudModel):
    def __init__(self, hyperparams):
        super(VNPointnet, self).__init__(hyperparams)
        self.model = "vn_pointnet"
        self.n_knn = hyperparams.n_knn
        self.normal_channel = hyperparams.normal_channel
        self.num_parts = hyperparams.num_parts
        self.pooling = hyperparams.pooling
        if self.normal_channel:
            channel = 6
        else:
            channel = 3

        self.conv_pos = VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 128 // 3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128 // 3, 128 // 3, dim=4, negative_slope=0.0)
        self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 512 // 3, dim=4, negative_slope=0.0)

        self.conv5 = VNLinear(512 // 3, 2048 // 3)
        self.bn5 = VNBatchNorm(2048 // 3, dim=4)

        self.std_feature = VNStdFeature(2048 // 3 * 2, dim=4, normalize_frame=False, negative_slope=0.0)

        if self.pooling == "max":
            self.pool = VNMaxPool(64 // 3)
        elif self.pooling == "mean":
            self.pool = mean_pool

        self.fstn = VNSTNkd(hyperparams, d=128 // 3)

        self.convs1 = torch.nn.Conv1d(9025, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, self.num_parts, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()

        point_cloud = point_cloud.unsqueeze(1)
        feat = get_graph_feature_cross(point_cloud, k=self.n_knn)
        point_cloud = self.conv_pos(feat)
        point_cloud = self.pool(point_cloud)

        out1 = self.conv1(point_cloud)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        net_global = self.fstn(out3).unsqueeze(-1).repeat(1, 1, 1, N)
        net_transformed = torch.cat((out3, net_global), 1)

        out4 = self.conv4(net_transformed)
        out5 = self.bn5(self.conv5(out4))

        out5_mean = out5.mean(dim=-1, keepdim=True).expand(out5.size())
        out5 = torch.cat((out5, out5_mean), 1)
        out5, trans = self.std_feature(out5)
        out5 = out5.view(B, -1, N)

        out_max = torch.max(out5, -1, keepdim=False)[0]

        out_max = torch.cat([out_max, label.squeeze(1)], 1)
        expand = out_max.view(-1, 2048 // 3 * 6 + 16, 1).repeat(1, 1, N)

        out1234 = torch.cat((out1, out2, out3, out4), dim=1)
        out1234 = torch.einsum("bijm,bjkm->bikm", out1234, trans).view(B, -1, N)

        concat = torch.cat([expand, out1234, out5], 1)

        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.num_parts), dim=-1)
        net = net.view(B, N, self.num_parts)  # [B, N, 50]

        return net
