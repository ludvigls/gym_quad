import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RadarCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) of dimension (N_sensors x 1)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, sensor_dim : int = 180, features_dim: int = 32, kernel_overlap : float = 0.05):
        super(RadarCNN, self).__init__(observation_space, features_dim=features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        # Adjust kernel size for sensor density. (Default 180 sensors with 0.05 overlap --> kernel covers 9 sensors.
        self.kernel_size = round(sensor_dim * kernel_overlap)
        self.kernel_size = self.kernel_size + 1 if self.kernel_size % 2 == 0 else self.kernel_size  # Make it odd sized
        self.padding = (self.kernel_size - 1) // 2
        self.cnn = nn.Sequential(
            # in_channels: stacked sensor distances (not velocities)
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=self.kernel_size, padding=self.padding,
                      padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=self.kernel_size, padding=self.padding,
                      padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=self.kernel_size, padding=self.padding,
                      padding_mode='circular', stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=self.kernel_size, padding=self.padding,
                      padding_mode='circular', stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        self.n_flatten = 0
        sample = th.as_tensor(observation_space.sample()).float()
        sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        with th.no_grad():
            print(RadarCNN)
            flatten = self.cnn(sample)
            self.n_flatten = flatten.shape[1]

        self.linear = nn.Sequential(nn.Linear(self.n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

    def get_features(self, observations: th.Tensor) -> list:
        feat = []
        out = observations
        for layer in self.cnn:
            out = layer(out)
            if not isinstance(layer, nn.ReLU):
                feat.append(out.detach().numpy())

        return feat

    def get_activations(self, observations: th.Tensor) -> list:
        feat = []
        out = observations
        for layer in self.cnn:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                feat.append(out)

        return feat


class NavigatioNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 6):
        super(NavigatioNN, self).__init__(observation_space, features_dim=features_dim)

        #self.passthrough = nn.Identity()
        self.passthrough = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        #shape = observations.shape
        #observations = observations.reshape(shape[0], shape[-1])
        return self.passthrough(observations)

class PerceptionNavigationExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) of dimension (1, 3, N_sensors)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, sensor_dim : int = 180, features_dim: int = 32, kernel_overlap : float = 0.05):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(PerceptionNavigationExtractor, self).__init__(observation_space, features_dim=1)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "perception":
                # Pass sensor readings through CNN
                extractors[key] = RadarCNN(subspace, sensor_dim=sensor_dim, features_dim=features_dim, kernel_overlap=kernel_overlap)
                total_concat_size += features_dim  # extractors[key].n_flatten
            elif key == "navigation":
                # Pass navigation features straight through to the MlpPolicy.
                extractors[key] = NavigatioNN(subspace, features_dim=subspace.shape[-1]) #nn.Identity()
                total_concat_size += subspace.shape[-1]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.ppo.policies import MlpPolicy
    from stable_baselines3 import PPO

    #### Test RadarCNN network circular 1D convolution:
    # Hyperparams
    n_sensors = 180
    kernel = 9
    padding = 4
    stride = 1

    ## Synthetic observation: (batch x channels x n_sensors)
    # Let obstacle be detected in the "edge" of the sensor array.
    # If circular padding works, should affect the outputs of the first <padding> elements
    obs = np.zeros((8, 3, n_sensors))
    obs[:, 0, :] = 1.0  # max distance
    obs[:, 0, -9:-1] = 0.3   # obstacle detected close in last 9 sensors
    obs[:, 1, :] = 0.0      # no obstacle
    obs[:, 2, :] = 0.0      # no obstacle
    obs = th.as_tensor(obs).float()

    ## Initialize convolutional layers (circular padding in all layers or just the first?)
    # First layer retains spatial structure,
    # includes circular padding to maintain the continuous radial structure of the RADAR,
    # and increased the feature-space dimensionality for extrapolation
    # (other padding types:)
    net1 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=kernel, padding=padding,
                      padding_mode='circular', stride=stride)
    # Second layer
    net2 = nn.Conv1d(in_channels=6, out_channels=3, kernel_size=kernel, padding=padding,
                      padding_mode='circular', stride=stride)
    net3 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=kernel, padding=padding,
                      padding_mode='circular', stride=2)
    net4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel, padding=padding,
                      padding_mode='circular', stride=2)

    flatten = nn.Flatten()
    act = nn.ReLU()
    #conv_weights = np.zeros(net1.weight.shape)

    #out1 = net1(obs)
    #out2 = net2(out1)
    #out3 = net3(out2)
    #out4 = net4(out3)
    out1 = act(net1(obs))
    out2 = act(net2(out1))
    out3 = act(net3(out2))
    out4 = act(net4(out3))

    feat = flatten(out4)


    ## Print shapes and characteritics of intermediate layer outputs
    obs = obs.detach().numpy()
    out1 = out1.detach().numpy()
    out2 = out2.detach().numpy()
    out3 = out3.detach().numpy()
    out4 = out4.detach().numpy()
    feat = feat.detach().numpy()

    def th2np_info(arr):
        #arr = tensor.detach().numpy()
        return "{:15.2f}{:15.2f}{:15.2f}{:15.2f}".format(arr.mean(), arr.std(), np.min(arr), np.max(arr))

    print("Observation",     obs.shape,  th2np_info(obs))
    print("First layer",     out1.shape, th2np_info(out1))
    print("Second layer",    out2.shape, th2np_info(out2))
    print("Third layer",     out3.shape, th2np_info(out3))
    print("Fourth layer",    out4.shape, th2np_info(out4))
    print("Output features", feat.shape, th2np_info(feat))

    ############### PLOTTING ################
    def radar_plot(features : list, names : list, kernels : list = [], title : str = "RadarCNN: intermediate layers visualization"):
        plt.style.use('ggplot')
        plt.rc('font', family='serif')
        # plt.rc('font', family='serif', serif='Times')
        # plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('axes', labelsize=8)
        plt.axis('scaled')
        def feat2radar(feat, avg=False):
            # Find length of feature vector
            n = feat.shape[-1] # number of activations differ between conv-layers
            feat = np.mean(feat, axis=0) if avg else feat[0] # average activations over batch or just select one

            # Find angles for each feature
            theta_d = 2 * np.pi / n  # Spatial spread according to the number of actications
            theta = np.array([(i + 1)*theta_d for i in range(n)]) # Angles for each activation

            # Hotfix: append first element of each list to connect the ends of the lines in the plot.
            theta = np.append(theta, theta[0])
            if len(feat.shape) > 1:
                _feat = []
                for ch, f in enumerate(feat):
                    ext = np.concatenate((f, [f[0]]))
                    _feat.append(ext)
            else:
                _feat = np.append(feat, feat[0])

            _feat = np.array(_feat)
            return theta, _feat  # Return angle positions & list of features.

        # Initialize polar plot, with zero located at the bottom (behind the vessel)
        fig, ax = plt.subplots(figsize=(11,11), subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location("S")
        ax.set_rmax(1)
        ax.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Less radial ticks
        ax.set_rlabel_position(22.5)  # Move radial labels away from plotted line
        ax.grid(True)
        ax.set_title(title, va='bottom')


        #features = [obs, out1, out2, out3, out4, feat]
        #names = ["obs", "out1", "out2", "out3", "out4", "feat"]
        linetypes = ["solid", 'dotted', 'dashed', 'dashdot', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5))]

        #CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
        #                  '#f781bf', '#a65628', '#984ea3',
        #                  '#999999', '#e41a1c', '#dede00']

        layer_color = {
            'obs' : '#377eb8',
            'conv1': '#ff7f00',
            'conv2': '#4daf4a',
            'conv3': '#f781bf',
            'conv4': '#a65628',
            'feat': '#984ea3',
        }

        for arr, layer in zip(features, names):
            angle, data = feat2radar(arr, avg=False)
            print("Layer", layer, ":")
            if len(data.shape) > 1:
                for ch, _d in enumerate(data):
                    print("\tch", ch, "shape:", _d.shape)
                    ax.plot(angle, _d, linestyle=linetypes[ch], color=layer_color[layer], label=layer+'_ch'+str(ch))
            else:
                ax.plot(angle, data, linestyle=linetypes[0], color=layer_color[layer], label=layer, linewidth=2)

        if kernels:
            _d_sensor_angle = 2*np.pi/n_sensors
            print("KERNELS", kernels)
            # Kernels: (layer x [width, padding, stride])
            for conv, kern in enumerate(kernels):
                size = kern[0]
                pad = kern[1]
                stride = kern[2]
                coverage = int(size + 2*pad)
                start_angle = (np.pi/180)*(conv*45 + 225)#/(2*np.pi) #- round(0.5 * coverage * _d_sensor_angle)
                #angles = [start_angle + _d_sensor_angle*i for i in range(coverage)]
                for step in range(3):
                    print("START_ABGLE", start_angle*(180/np.pi))
                    alpha = 0.5 - 0.1*step
                    ax.bar(start_angle, height=0.05, bottom=0.9, width=(coverage/n_sensors) * (2*np.pi), color=(0.7,0.7,0.7), alpha=alpha)
                    start_angle += stride * _d_sensor_angle

        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
        plt.tight_layout()
        plt.show()

    ###### PLOT FOR SYNTHETIC CONVNET #####
    features = [obs, out1, out2, out3, out4, feat]
    names = ["obs", "conv1", "conv2", "conv3", "conv4", "feat"]
    radar_plot(features, names)
    print('plot1')


    ################ NOW DO IT WITH NEW RadarCNN INSTANCE #########
    def get_features_from_new_net(obs=None):
        import gym
        obs_tensor = th.as_tensor(obs).float()
        perception_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1, 180),
            dtype=np.float32
        )
        net = RadarCNN(perception_space, sensor_dim=180, features_dim=32, kernel_overlap=0.05)
        features = net.get_features(obs_tensor)
        return features

    obs = obs[:,0,:]
    obs = obs[:,np.newaxis,:]
    features = get_features_from_new_net(obs)
    features.insert(0, obs)
    names = ["obs", "conv1", "conv2", "conv3", "feat"]
    kernels = [[9,4,1],[9,4,1],[9,4,2],[9,4,2]]
    title = "get_features_from_new_net"
    radar_plot(features, names, kernels=kernels, title=title)
    print("plot2")
    ################ NOW DO IT WITH A TRAINED MODEL ###############
    ## Load existing convnet
    def get_features_from_existing_net(obs=None):
        import gym_auv
        algo = PPO
        path = "radarCNN_example_Network750k.pkl"

        # Load model
        model = algo.load(path)
        # Get RadarCNN feature extractor model
        extractor = model.policy.features_extractor.extractors["perception"]
        # Calculate features
        obs_cuda = th.as_tensor(obs).float().cuda()

        #features = extractor(obs_cuda).cpu().detach().numpy()
        int_feat = extractor.get_features(obs_cuda)

        #_feat = []
        #out = obs_cuda
        #for layer in extractor.cnn:
        #    out = layer(out)
        #    if not isinstance(layer, nn.ReLU):
        #        _feat.append(out.cpu().detach().numpy())

        #for layer in extractor.linear:
        #    out = layer(out)
        #    if not isinstance(layer, nn.ReLU):
        #        _feat.append(out.cpu().detach().numpy())

        return int_feat #_feat

    features = get_features_from_existing_net(obs)
    features.insert(0, obs)
    names = ["obs", "conv1", "conv2", "conv3", "conv4", "feat"]
    title = "get_features_from_existing_net"
    radar_plot(features, names, title)
    print("plot3")

