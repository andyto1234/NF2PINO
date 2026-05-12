import numpy as np
import torch
from astropy import units as u
from torch import nn

from nf2.data.util import cartesian_to_spherical, spherical_to_cartesian


class Swish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class RadialTransformModel(nn.Module):

    def __init__(self, in_coords, dim, positional_encoding=True, ds_ids=[]):
        super().__init__()
        if positional_encoding:
            posenc = GaussianPositionalEncoding(num_freqs=20,
                                                d_input=in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            self.d_in = nn.Linear(in_coords, dim)
        lin = [nn.Linear(dim, dim) for _ in range(4)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, 1)
        self.activation = Sine()
        self.ds_ids = ds_ids
        self.observer_transformer = ObserverTransformer()

    def forward(self, batch):
        for ds_id in self.ds_ids:
            transformed_coords = self.transform(batch[ds_id]['coords'], batch[ds_id]['obs_coords'],
                                                batch[ds_id]['height_range'])
            batch[ds_id]['coords'] = transformed_coords
            batch[ds_id]['original_coords'] = coords

    def transform(self, coords, obs_coords, height_range, **kwargs):
        coords = self.observer_transformer.transform(coords, obs_coords)

        x = self.activation(self.d_in(coords))
        for l in self.linear_layers:
            x = self.activation(l(x))
        z_coords = torch.sigmoid(self.d_out(x)) * (height_range[:, 1:2] - height_range[:, 0:1]) + height_range[:, 0:1]
        output_coords = torch.cat([coords[:, :2], z_coords], -1)

        output_coords = self.observer_transformer.inverse_transform(output_coords, obs_coords)

        return output_coords


class GenericModel(nn.Module):

    def __init__(self, in_coords, out_coords, dim=256, n_layers=8, encoding=None, activation='sine'):
        super().__init__()
        if encoding is None or encoding == 'none':
            self.d_in = nn.Linear(in_coords, dim)
        elif encoding == 'positional':
            posenc = PositionalEncoding(20, in_coords)
            d_in = nn.Linear(in_coords * 40, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == 'gaussian':
            posenc = GaussianPositionalEncoding(20, in_coords)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            raise NotImplementedError(f'Unknown encoding {encoding}')
        lin = [nn.Linear(dim, dim) for _ in range(n_layers)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, out_coords)
        activation_mapping = {'relu': nn.ReLU, 'swish': Swish, 'tanh': nn.Tanh, 'sine': Sine}
        activation_f = activation_mapping[activation]
        self.in_activation = activation_f()
        self.activations = nn.ModuleList([activation_f() for _ in range(n_layers)])

    def forward(self, x):
        x = self.in_activation(self.d_in(x))
        for l, a in zip(self.linear_layers, self.activations):
            x = a(l(x))
        x = self.d_out(x)
        return x


class BModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(3, 3, **kwargs)

    def forward(self, coords, compute_jacobian=True):
        b = super().forward(coords)
        out_dict = {'b': b}
        if compute_jacobian:
            jac_matrix = jacobian(b, coords)
            out_dict['jac_matrix'] = jac_matrix
        return out_dict


class VectorPotentialModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(3, 3, **kwargs)

    def forward(self, coords, compute_jacobian=True):
        a = super().forward(coords)
        #
        jac_matrix = jacobian(a, coords)
        dAy_dx = jac_matrix[:, 1, 0]
        dAz_dx = jac_matrix[:, 2, 0]
        dAx_dy = jac_matrix[:, 0, 1]
        dAz_dy = jac_matrix[:, 2, 1]
        dAx_dz = jac_matrix[:, 0, 2]
        dAy_dz = jac_matrix[:, 1, 2]
        rot_x = dAz_dy - dAy_dz
        rot_y = dAx_dz - dAz_dx
        rot_z = dAy_dx - dAx_dy
        b = torch.stack([rot_x, rot_y, rot_z], -1)
        out_dict = {'b': b, 'a': a}
        #
        if compute_jacobian:
            jac_matrix = jacobian(b, coords)
            out_dict['jac_matrix'] = jac_matrix
        #
        return out_dict


class PressureModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(3, 1, **kwargs)
        self.softplus = nn.Softplus()

    def forward(self, x):
        p = super().forward(x)
        p = 10 ** p
        return {'p': p}


class MagnetoStaticModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(3, 4)

    def forward(self, coords, compute_jacobian=True):
        model_out = super().forward(coords)
        b = model_out[:, :3]
        p = 10 ** model_out[:, 3:]
        out_dict = {'b': b, 'p': p}
        if compute_jacobian:
            jac_matrix = jacobian(model_out, coords)
            out_dict['jac_matrix'] = jac_matrix
        return out_dict


class MagnetoStaticModelV2(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.b_model = VectorPotentialModel(**kwargs)
        self.p_model = PressureModel(**kwargs)

    def forward(self, coords, compute_jacobian=True):
        b_dict = self.b_model(coords)
        p_dict = self.p_model(coords)
        out_dict = {**b_dict, **p_dict}
        if compute_jacobian:
            b = b_dict['b']
            jac_b_matrix = jacobian(b, coords)
            p = p_dict['p']
            jac_p_matrix = jacobian(p, coords)
            jac_matrix = torch.cat([jac_b_matrix, jac_p_matrix], -2)
            out_dict['jac_matrix'] = jac_matrix

        return out_dict


class PositionalEncoding(nn.Module):

    def __init__(self, num_freqs, in_features, max_freq=8):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq, num_freqs)
        self.frequencies = nn.Parameter(frequencies[None, :, None], requires_grad=False)
        self.d_output = in_features * (num_freqs * 2)

    def forward(self, x):
        encoded = x[:, None, :] * torch.pi * self.frequencies
        encoded = encoded.reshape(x.shape[0], -1)
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded

class GaussianPositionalEncoding(nn.Module):

    def __init__(self, num_freqs, d_input, scale=1.):
        super().__init__()
        frequencies = torch.randn(num_freqs, d_input) * scale
        self.frequencies = nn.Parameter(frequencies[None], requires_grad=False)
        self.d_output = d_input * (num_freqs * 2 + 1)

    def forward(self, x):
        encoded = x[:, None, :] * self.frequencies
        encoded = encoded.reshape(x.shape[0], -1)
        encoded = torch.cat([x, torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded


class ObserverTransformer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, coords, obs_coord):
        return self.transform(coords, obs_coord)

    def transform(self, coords, obs_coord):
        # transform coords to observer frame
        coords = coords - obs_coord
        coords = cartesian_to_spherical(coords, f=torch)
        return coords

    def inverse_transform(self, coords, obs_coord):
        # transform coords to solar frame
        coords = spherical_to_cartesian(coords, f=torch)
        coords = coords + obs_coord
        return coords


def image_to_spherical_matrix(lon, lat, latc, lonc, pAng, sin=np.sin, cos=np.cos):
    a11 = -sin(latc) * sin(pAng) * sin(lon - lonc) + cos(pAng) * cos(lon - lonc)
    a12 = sin(latc) * cos(pAng) * sin(lon - lonc) + sin(pAng) * cos(lon - lonc)
    a13 = -cos(latc) * sin(lon - lonc)
    a21 = -sin(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) - cos(lat) * cos(
        latc) * sin(pAng)
    a22 = sin(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) + cos(lat) * cos(
        latc) * cos(pAng)
    a23 = -cos(latc) * sin(lat) * cos(lon - lonc) + sin(latc) * cos(lat)
    a31 = cos(lat) * (sin(latc) * sin(pAng) * cos(lon - lonc) + cos(pAng) * sin(lon - lonc)) - sin(lat) * cos(
        latc) * sin(pAng)
    a32 = -cos(lat) * (sin(latc) * cos(pAng) * cos(lon - lonc) - sin(pAng) * sin(lon - lonc)) + sin(lat) * cos(
        latc) * cos(pAng)
    a33 = cos(lat) * cos(latc) * cos(lon - lonc) + sin(lat) * sin(latc)

    # a_matrix = np.stack([a11, a12, a13, a21, a22, a23, a31, a32, a33], axis=-1)
    a_matrix = np.stack([a31, a32, a33, a21, a22, a23, a11, a12, a13], axis=-1)
    a_matrix = a_matrix.reshape((*a_matrix.shape[:-1], 3, 3))
    return a_matrix


def calculate_current(b, coords, jac_matrix=None):
    jac_matrix = jacobian(b, coords) if jac_matrix is None else jac_matrix
    j = calculate_current_from_jacobian(jac_matrix)
    return j


def calculate_current_from_jacobian(jac_matrix, f=torch):
    dBx_dx = jac_matrix[..., 0, 0]
    dBy_dx = jac_matrix[..., 1, 0]
    dBz_dx = jac_matrix[..., 2, 0]
    dBx_dy = jac_matrix[..., 0, 1]
    dBy_dy = jac_matrix[..., 1, 1]
    dBz_dy = jac_matrix[..., 2, 1]
    dBx_dz = jac_matrix[..., 0, 2]
    dBy_dz = jac_matrix[..., 1, 2]
    dBz_dz = jac_matrix[..., 2, 2]
    #
    rot_x = dBz_dy - dBy_dz
    rot_y = dBx_dz - dBz_dx
    rot_z = dBy_dx - dBx_dy
    #
    j = f.stack([rot_x, rot_y, rot_z], -1)
    return j


def jacobian(output, coords):
    jac_matrix = [torch.autograd.grad(output[:, i], coords,
                                      grad_outputs=torch.ones_like(output[:, i]).to(output),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
                  for i in range(output.shape[1])]
    jac_matrix = torch.stack(jac_matrix, dim=1)
    return jac_matrix
