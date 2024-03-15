"""
This module defines a base class for canonicalization and its subclasses for different types of canonicalization methods.

Canonicalization is a process that transforms the input data into a canonical (standard) form.
This can be cheap alternative to building equivariant models as it can be used to transform the input data into a canonical form and then use a standard model to make predictions.
Canonicalizarion allows you to use any existing arcitecture (even pre-trained ones) for your task without having to worry about equivariance.

The module contains the following classes:

- `BaseCanonicalization`: This is an abstract base class that defines the interface for all canonicalization methods.

- `IdentityCanonicalization`: This class represents an identity canonicalization method, which is a no-op; it doesn't change the input data.

- `DiscreteGroupCanonicalization`: This class represents a discrete group canonicalization method, which transforms the input data into a canonical form using a discrete group.

- `ContinuousGroupCanonicalization`: This class represents a continuous group canonicalization method, which transforms the input data into a canonical form using a continuous group.

Each class has methods to perform the canonicalization, invert it, and calculate the prior regularization loss and identity metric.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# Base skeleton for the canonicalization class
# DiscreteGroupCanonicalization and ContinuousGroupCanonicalization will inherit from this class


class BaseCanonicalization(torch.nn.Module):
    """
    This is the base class for canonicalization.

    This class is used as a base for all canonicalization methods.
    Subclasses should implement the canonicalize method to define the specific canonicalization process.

    """

    def __init__(self, canonicalization_network: torch.nn.Module):
        super().__init__()
        self.canonicalization_network = canonicalization_network
        self.canonicalization_info_dict: Dict[str, torch.Tensor] = {}

    def forward(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Forward method for the canonicalization which takes the input data and returns the canonicalized version of the data

        Args:
            x: input data
            targets: (optional) additional targets that need to be canonicalized,
                    such as boxes for promptable instance segmentation
            **kwargs: additional arguments

        Returns:
            canonicalized_x: canonicalized version of the input data

        """
        # call the canonicalize method to obtain canonicalized version of the input data
        return self.canonicalize(x, targets, **kwargs)

    def canonicalize(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        This method takes an input data with, optionally, targets that need to be canonicalized

        Args:
            x: input data
            targets: (optional) additional targets that need to be canonicalized,
                    such as boxes for promptable instance segmentation
            **kwargs: additional arguments

        Returns:
            the canonicalized version of the data and targets
        """
        raise NotImplementedError()

    def invert_canonicalization(
        self, x_canonicalized_out: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        This method takes the output of the canonicalized data and returns the output for the original data orientation

        Args:
            canonicalized_outputs: output of the prediction network for canonicalized data

        Returns:
            outputs: output of the prediction network for the original data orientation,
                by using the group element used to canonicalize the original data

        """
        raise NotImplementedError()


class IdentityCanonicalization(BaseCanonicalization):
    """
    This class represents an identity canonicalization method.

    Identity canonicalization is a no-op; it doesn't change the input data. It's useful as a default or placeholder
    when no other canonicalization method is specified.

    Attributes:
        canonicalization_network (torch.nn.Module): The network used for canonicalization. Defaults to torch.nn.Identity.

    Methods:
        __init__: Initializes the IdentityCanonicalization instance.
        canonicalize: Canonicalizes the input data. In this class, it returns the input data unchanged.
    """

    def __init__(self, canonicalization_network: torch.nn.Module = torch.nn.Identity()):
        """
        Initializes the IdentityCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module, optional): The network used for canonicalization. Defaults to torch.nn.Identity.
        """
        super().__init__(canonicalization_network)

    def canonicalize(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Canonicalize the input data.

        This method takes the input data and returns it unchanged, along with the targets if provided.
        It's a no-op in the IdentityCanonicalization class.

        Args:
            x: The input data.
            targets: (Optional) Additional targets that need to be canonicalized.
            **kwargs: Additional arguments.

        Returns:
            A tuple containing the unchanged input data and targets if targets are provided,
            otherwise just the unchanged input data.
        """
        if targets:
            return x, targets
        return x

    def invert_canonicalization(
        self, x_canonicalized_out: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Inverts the canonicalization.

        For the IdentityCanonicalization class, this is a no-op and returns the input unchanged.

        Args:
            x_canonicalized_out (torch.Tensor): The canonicalized output.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The unchanged x_canonicalized_out.
        """
        return x_canonicalized_out

    def get_prior_regularization_loss(self) -> torch.Tensor:
        """
        Gets the prior regularization loss.

        For the IdentityCanonicalization class, this is always 0.

        Returns:
            torch.Tensor: A tensor containing the value 0.
        """
        return torch.tensor(0.0)

    def get_identity_metric(self) -> torch.Tensor:
        """
        Gets the identity metric.

        For the IdentityCanonicalization class, this is always 1.

        Returns:
            torch.Tensor: A tensor containing the value 1.
        """
        return torch.tensor(1.0)


class DiscreteGroupCanonicalization(BaseCanonicalization):
    """
    This class represents a discrete group canonicalization method.

    Discrete group canonicalization is a method that transforms the input data into a canonical form using a discrete group. This class is a subclass of the BaseCanonicalization class and overrides its methods to provide the functionality for discrete group canonicalization.

    Attributes:
        canonicalization_network (torch.nn.Module): The network used for canonicalization.
        beta (float): A parameter for the softmax function. Defaults to 1.0.
        gradient_trick (str): The method used for backpropagation through the discrete operation. Defaults to "straight_through".

    Methods:
        __init__: Initializes the DiscreteGroupCanonicalization instance.
        groupactivations_to_groupelementonehot: Converts group activations to one-hot encoded group elements in a differentiable manner.
        canonicalize: Canonicalizes the input data.
        invert_canonicalization: Inverts the canonicalization.
        get_prior_regularization_loss: Gets the prior regularization loss.
        get_identity_metric: Gets the identity metric.

    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        beta: float = 1.0,
        gradient_trick: str = "straight_through",
    ):
        """
        Initializes the DiscreteGroupCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The network used for canonicalization.
            beta (float, optional): A parameter for the softmax function. Defaults to 1.0.
            gradient_trick (str, optional): The method used for backpropagation through the discrete operation. Defaults to "straight_through".
        """
        super().__init__(canonicalization_network)
        self.beta = beta
        self.gradient_trick = gradient_trick

    def groupactivations_to_groupelementonehot(
        self, group_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Converts group activations to one-hot encoded group elements in a differentiable manner.

        Args:
            group_activations (torch.Tensor): The activations for each group element.

        Returns:
            torch.Tensor: The one-hot encoding of the group elements.
        """
        group_activations_one_hot = torch.nn.functional.one_hot(
            torch.argmax(group_activations, dim=-1), self.num_group
        ).float()
        group_activations_soft = torch.nn.functional.softmax(
            self.beta * group_activations, dim=-1
        )
        if self.gradient_trick == "straight_through":
            if self.training:
                group_element_onehot = (
                    group_activations_one_hot
                    + group_activations_soft
                    - group_activations_soft.detach()
                )
            else:
                group_element_onehot = group_activations_one_hot
        elif self.gradient_trick == "gumbel_softmax":
            group_element_onehot = torch.nn.functional.gumbel_softmax(
                group_activations, tau=1, hard=True
            )
        else:
            raise ValueError(f"Gradient trick {self.gradient_trick} not implemented")

        # return the group element one hot encoding
        return group_element_onehot

    def canonicalize(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Canonicalizes the input data.

        Args:
            x (torch.Tensor): The input data.
            targets (List, optional): Additional targets that need to be canonicalized.
            **kwargs: Additional arguments.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List]]: The canonicalized input data and targets.
            Simultaneously, it updates a dictionary containing the information about the canonicalization.
        """
        raise NotImplementedError()

    def invert_canonicalization(
        self, x_canonicalized_out: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Inverts the canonicalization.

        Args:
            x_canonicalized_out (torch.Tensor): The canonicalized output.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The output for the original data orientation.
        """
        raise NotImplementedError()

    def get_prior_regularization_loss(self) -> torch.Tensor:
        """
        Gets the prior regularization loss.

        Returns:
            torch.Tensor: The prior regularization loss.
        """
        group_activations = self.canonicalization_info_dict["group_activations"]
        dataset_prior = torch.zeros((group_activations.shape[0],), dtype=torch.long).to(
            self.device
        )
        return torch.nn.CrossEntropyLoss()(group_activations, dataset_prior)

    def get_identity_metric(self) -> torch.Tensor:
        """
        Gets the identity metric.

        Returns:
            torch.Tensor: The identity metric.
        """
        group_activations = self.canonicalization_info_dict["group_activations"]
        return (group_activations.argmax(dim=-1) == 0).float().mean()


class ContinuousGroupCanonicalization(BaseCanonicalization):
    """
    This class represents a continuous group canonicalization method.

    Continuous group canonicalization is a method that transforms the input data into a canonical form using a continuous group. This class is a subclass of the BaseCanonicalization class and overrides its methods to provide the functionality for continuous group canonicalization.

    Attributes:
        canonicalization_network (torch.nn.Module): The network used for canonicalization.
        beta (float): A parameter for the softmax function. Defaults to 1.0.

    Methods:
        __init__: Initializes the ContinuousGroupCanonicalization instance.
        canonicalizationnetworkout_to_groupelement: Converts the output of the canonicalization network to a group element in a differentiable manner.
        canonicalize: Canonicalizes the input data.
        invert_canonicalization: Inverts the canonicalization.
        get_prior_regularization_loss: Gets the prior regularization loss.
        get_identity_metric: Gets the identity metric.
    """

    def __init__(self, canonicalization_network: torch.nn.Module, beta: float = 1.0):
        """
        Initializes the ContinuousGroupCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The network used for canonicalization.
            beta (float, optional): A parameter for the softmax function. Defaults to 1.0.
        """
        super().__init__(canonicalization_network)
        self.beta = beta

    def canonicalizationnetworkout_to_groupelement(
        self, group_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Converts the output of the canonicalization network to a group element in a differentiable manner.

        Args:
            group_activations (torch.Tensor): The activations for each group element.

        Returns:
            torch.Tensor: The group element.
        """
        raise NotImplementedError()

    def canonicalize(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Canonicalizes the input data.

        Args:
            x (torch.Tensor): The input data.
            targets (List, optional): Additional targets that need to be canonicalized.
            **kwargs: Additional arguments.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List]]: The canonicalized input data and targets.
            Simultaneously, it updates a dictionary containing the information about the canonicalization.
        """
        raise NotImplementedError()

    def invert_canonicalization(
        self, x_canonicalized_out: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Inverts the canonicalization.

        Args:
            x_canonicalized_out (torch.Tensor): The canonicalized output.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: The output for the original data orientation.
        """
        raise NotImplementedError()

    def get_prior_regularization_loss(self) -> torch.Tensor:
        """
        Gets the prior regularization loss.

        The prior regularization loss is calculated as the mean squared error between the group element matrix representation and the identity matrix.

        Returns:
            torch.Tensor: The prior regularization loss.
        """
        group_elements_rep = self.canonicalization_info_dict[
            "group_element_matrix_representation"
        ]  # shape: (batch_size, group_rep_dim, group_rep_dim)
        # Set the dataset prior to identity matrix of size group_rep_dim and repeat it for batch_size
        dataset_prior = (
            torch.eye(group_elements_rep.shape[-1])
            .repeat(group_elements_rep.shape[0], 1, 1)
            .to(self.device)
        )
        return torch.nn.MSELoss()(group_elements_rep, dataset_prior)

    def get_identity_metric(self) -> torch.Tensor:
        """
        Gets the identity metric.

        The identity metric is calculated as 1 minus the mean of the mean squared error between the group element matrix representation and the identity matrix.

        Returns:
            torch.Tensor: The identity metric.
        """
        group_elements_rep = self.canonicalization_info_dict[
            "group_element_matrix_representation"
        ]
        identity_element = (
            torch.eye(group_elements_rep.shape[-1])
            .repeat(group_elements_rep.shape[0], 1, 1)
            .to(self.device)
        )
        return (
            1.0
            - torch.nn.functional.mse_loss(group_elements_rep, identity_element).mean()
        )
