#!/usr/bin/env python

import torch
import numpy
import MDAnalysis as mda
import itertools
from MDAnalysis.analysis.distances import dist

# neural network residual unit (credits: H. Jung)
class _PreActivationResidualUnit(torch.nn.Module):
    """
    Full pre-activation residual unit.

    Literature:
    'Identity Mappings in Deep Residual Networks' by He et al (arXiv:1603.05027)

    """

    def __init__(self, n_units, n_skip=4,
                 activation=torch.nn.functional.elu, norm_layer=None):
        """
        Initialize PreActivationResidualUnit.

        n_units - number of units per layer
        n_skip - number of layers to skip with the identity connection
        activation - activation function class
        norm_layer - normalization layer class
        """
        super().__init__()
        self.n_out = n_units
        self.call_kwargs = {'n_units': n_units,
                            'n_skip': n_skip,
                            'activation': activation,
                            'norm_layer': norm_layer,
                            }  # I think we do not need this here...
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(n_units, n_units) for _ in range(n_skip)])
        if norm_layer is None:
            # TODO: is this really what we want?!
            norm_layer = torch.nn.Identity
        self.norm_layers = torch.nn.ModuleList([norm_layer(n_units)
                                          for _ in range(n_skip)])
        # TODO: do we want to be able to use different activations?
        # i.e. should we use a list of activation functions?
        self.activation = activation
        self.reset_parameters()

    def forward(self, x):
        identity = x
        for lay, norm in zip(self.layers, self.norm_layers):
            x = lay(self.activation(norm(x)))
        x = x + identity
        return x

    def reset_parameters(self):
        for lay in self.norm_layers:
            # TODO: how to best reset/initialize the possible
            #       (batch)normalization layers?
            reset_func = getattr(lay, "reset_parameters", None)
            if reset_func is not None:
                lay.reset_parameters()
        # TODO: proper initialization depends on the activation function
        #       this works only for activations that result in zero mean
        for lay in self.layers[:-1]:
            # these are all linear layers, their default initialization is
            # xavier uniform
            lay.reset_parameters()
        # initialize the last linear layer weights and biases to zero
        # see e.g. 'fixup initialization'
        torch.nn.init.zeros_(self.layers[-1].weight)
        if self.layers[-1].bias is not None:
            torch.nn.init.zeros_(self.layers[-1].bias)


# class defining the neural network
# attention! it must be last in alphabetical order
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_kwargs = {}
        # layers & activations
        self.layer1 = torch.nn.Linear(2064      , 2048 * 4  )
        self.layer2 = torch.nn.Linear(2048 * 4  , 2048      )
        self.layer3 = torch.nn.Linear(2048      , 2048 // 4 )
        self.layer4 = torch.nn.Linear(2048 // 4 , 2048 // 16)
        self.layer5 = torch.nn.Linear(2048 // 16, 1         )
        self.resnet1 = _PreActivationResidualUnit(n_units=2048 // 16)
        self.resnet2 = _PreActivationResidualUnit(n_units=2048 // 16)
        self.resnet3 = _PreActivationResidualUnit(n_units=2048 // 16)
        self.resnet4 = _PreActivationResidualUnit(n_units=2048 // 16)
        self.activation1 = torch.nn.ELU()
        self.activation2 = torch.nn.ELU()
        self.activation3 = torch.nn.ELU()
        self.activation4 = torch.nn.ELU()
        self.reset_parameters()
    def forward(self, x):                    #           ii         input
        x = self.activation1(self.layer1(x)) #  ................... 2048 * 4
        x = self.activation2(self.layer2(x)) #     .............    2048
        x = self.activation3(self.layer3(x)) #         .....        2048 / 4
        x = self.activation4(self.layer4(x)) #          ...         2048 // 16
        x = self.resnet1(x)                  #          ...         resid 1
        x = self.resnet2(x)                  #          ...         resid 2
        x = self.resnet3(x)                  #          ...         resid 3
        x = self.resnet4(x)                  #          ...         resid 4
        x = self.layer5(x)  # final result (real number) o          1
        return x
    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()
        self.layer4.reset_parameters()
        self.layer5.reset_parameters()
        self.resnet1.reset_parameters()
        self.resnet2.reset_parameters()
        self.resnet3.reset_parameters()
        self.resnet4.reset_parameters()


def atom_select(universe):
    """
    Sample featurization, Overwrite/monkey-patch.
    This is for chignolin, selecting heavy atoms that are not adjacent (resid > 3).

    """
    heavy = universe.select_atoms('protein and not name H*')
    pair_indices = numpy.array(
        [(i,j) for (i,j) in itertools.combinations(heavy, 2)
            if abs(i.resid - j.resid) > 3])
    ag1 = mda.core.groups.AtomGroup(pair_indices[:, 0])
    ag2 = mda.core.groups.AtomGroup(pair_indices[:, 1])

    return ag1, ag2


def featurize(ag1, ag2, box=None, dmin_file=None, dmax_file= None):
    """
    
    Parameters
    ----------
    ag1: AtomGroup
        First MDAnalysis atom group

    ag2: AtomGroup
        Second MDAnalysis atom group

    box: nump.array
        Simulation box so that distances are calculated using periodic boundaries

    dmin_file: str
        Path to min distance numpy file for numpy to load.

    dmax_file: str
        Path to max distance numpy file for numpy to load.

    Returns
    -------
    distance matrix: numpy.ndarray
        Pairwise distance matrix.
    
    """
    # Min/Max normalization (if provided), then calculate pairwise dist.
    if dmin_file and dmax_file:
        dmin = numpy.load(dmin_file, allow_pickle=True)
        dmax = numpy.load(dmax_file, allow_pickle=True)

        # Return Normalized distance
        return  ((dist(ag1, ag2, box=box)[2, :])/10 - dmin[None, :]) / (dmax - dmin)[None, :]
    else:
        return (dist(ag1, ag2, box=box)/10)[2, :]



if __name__ == '__main__':
    # Create universe
    u = mda.Universe('seg.gro', 'seg.trr')
    ag1, ag2 = atom_select(u)

    model = Network() # Create model

    # map_location will depend on what system you use. But mapping to cpu probably the most compatible
    state_dict = torch.load('model000250.pth', map_location=torch.device('cpu')) 
    
    model.load_state_dict(state_dict) # Load in parameters
    model.eval() # Turn training mode off

    # Compute descriptors
    # Need to convert to float32 for compatibility with the model
    descriptors = numpy.float32(featurize(ag1, ag2, box=u.trajectory[0].dimensions, dmin_file='dmin.npy', dmax_file='dmax.npy'))
    # Input descriptors into NN
    with torch.no_grad():
        output = model(torch.as_tensor(descriptors))

    committor = torch.special.expit(output).detach().numpy().ravel()

    # Output into file...
    #numpy.save('nn_output.npy', committor)
    numpy.savetxt('nn_output.dat', committor)
