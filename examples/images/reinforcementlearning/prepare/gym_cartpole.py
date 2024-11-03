import gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class CartpoleWrapper(gym.Wrapper):
    def __init__(self, env, env_hyperparams):
        """Initialize the wrapper for the CartPole environment to preprocess images.

        Args:
            env (gym.Env): The Gym environment to wrap.
            env_hyperparams (dict): Dictionary containing settings for image preprocessing,
                                    such as resize dimensions and whether to apply grayscale.
        """
        super().__init__(env)
        self.env = env
        self.num_actions = env.action_space.n
        self.state_shape = env.observation_space.shape

        # Base transformations that are always applied
        transformations = [
            T.ToPILImage(),
            T.Resize(env_hyperparams["resize_pixels"], interpolation=Image.BICUBIC),
        ]

        # Conditional grayscale transformation
        if env_hyperparams["grayscale"]:
            transformations.append(T.Grayscale())

        # Final transformation to tensor
        transformations.append(T.ToTensor())

        # Compose all transformations into a single callable object
        self.resize = T.Compose(transformations)

    def get_cart_location(self, screen_width):
        """Calculate the cart's location on the screen for cropping.

        Args:
            screen_width (int): The width of the screen from the environment.

        Returns:
            int: The pixel location of the center of the cart.
        """
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # Middle of the cart

    def get_screen(self):
        """Capture, process, and crop the environment's screen.

        Transforms the screen into a format suitable for input to a neural network:
        crops, downsamples, converts to grayscale, and rescales.

        Returns:
            torch.Tensor: The processed screen tensor ready for model input.
        """
        # Capture screen from the environment
        screen = self.env.render().transpose((2, 0, 1))  # CHW format
        _, screen_height, screen_width = screen.shape

        # Crop the vertical dimension to focus on the main area of interest
        screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]

        # Define the width of the cropped area around the cart
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)

        # Calculate the horizontal slice range to center crop around the cart
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(
                cart_location - view_width // 2, cart_location + view_width // 2
            )

        # Apply the calculated slice to crop horizontally
        screen = screen[:, :, slice_range]

        # Normalize, convert to tensor, resize, and add a batch dimension
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
        screen = torch.from_numpy(screen)
        return self.resize(screen).unsqueeze(0)

    def step(self, action):
        """Apply an action to the environment, returning the processed screen, reward, done, and info."""
        return self.env.step(action)

    def reset(self):
        """Reset the environment and return the initial processed screen."""
        self.env.reset()
