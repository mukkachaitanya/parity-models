import torch
from torch import nn

from coders.coder import Coder, Decoder
import util.util
import pdb


class LinearEncoder(Coder):
    """
    Constructs parities as a linear combination of inputs according to the
    ``coefficients``.
    """
    def __init__(self, ec_k, ec_r, in_dim, coefficients):
        super().__init__(ec_k, ec_r, in_dim)
        self.coefficients = util.util.try_cuda(torch.FloatTensor(coefficients))
        self.coefficients = self.coefficients.unsqueeze(0)

    def forward(self, in_data):
        while len(self.coefficients.size()) < len(in_data.size()):
            self.coefficients = self.coefficients.unsqueeze(-1)

        coeff = self.coefficients.expand_as(in_data)
        return torch.sum(in_data * coeff, dim=1).view(in_data.size(0), self.num_out, -1)


class LinearDecoder(Coder):
    """
    Reconstructs unavailable outputs according to the linear combination
    described by ``coefficients``.
    Only reconstructs a single unavailable output.
    Example:
        Suppose the coefficents are [ a, b, c ]
        Then parity will be constructed for inputs X1, X2, and X3 as:
            P = aX1 + bX2 + cX3
        The corresonding reconstruction operations (again, assuming only one
        unavailable output) will be:
            X1 = (P - bX2 - cX3) / a
            X2 = (P - aX1 - cX3) / b
            X3 = (P - aX1 - bX2) / c
    """
    def __init__(self, ec_k, ec_r, in_dim, coefficients):
        super().__init__(ec_k, ec_r, in_dim)
        self.coefficients = torch.FloatTensor(coefficients)

        self.label_coefficients = self.coefficients.unsqueeze(0)
        self.inv_coefficients = torch.ones(self.label_coefficients.size()) / self.label_coefficients
        self.inv_coefficients = util.util.try_cuda(self.inv_coefficients)
        self.label_coefficients = util.util.try_cuda(self.label_coefficients)

    def forward(self, in_data):
        """
        Computes a reconstruction of an unavailable output of the form:
            Y1 = (P - bY2 - cY3) / a
        Where Y1 is the unavailable output, Y2 and Y3 are the available
        outputs, and P is the result of inference over the parity:
            P = aX1 + bX2 + cX3
        For inputs X1, X2, and X3.
        Parameters
        ----------
        in_data : (torch.autograd.Variable)
            Dimensionality (batch_size * k, k+1, label_size). There should be
            `k` repeats of the same `batch_size` labels.
        """
        while len(self.label_coefficients.size()) < len(in_data.size()):
            self.label_coefficients = self.label_coefficients.unsqueeze(-1)
            self.inv_coefficients = self.inv_coefficients.unsqueeze(-1)

        coeffs = self.label_coefficients.expand_as(in_data[:,:-1])

        # Compute the P - bY2 - cY3 part
        out = in_data[:, -1] - torch.sum(in_data[:,:-1] * coeffs, dim=1)
        out = out.view(-1, self.num_in, out.size(-1))

        # Divide by coefficients used in original encoding
        inv_coeffs = self.inv_coefficients.expand_as(out)
        out = out * inv_coeffs
        out = out.view(-1, 1, out.size(-1))
        out = out.repeat(1, self.num_in, 1)
        return out

    def combine_labels(self, in_data):
        while len(self.label_coefficients.size()) < len(in_data.size()):
            self.label_coefficients = self.label_coefficients.unsqueeze(-1)
            self.inv_coefficients = self.inv_coefficients.unsqueeze(-1)

        coeff = self.label_coefficients.expand_as(in_data)
        out = torch.sum(in_data * coeff, dim=1)
        return out


if __name__ == "__main__":
    pdb.set_trace()
    coder = LinearDecoder(2, 1, 10, [0.5, 0.5])
    labels = torch.autograd.Variable(torch.FloatTensor([
        [
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]
        ],
        [
            [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
            [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.]
        ]
        ]))
    combined_labels = coder.combine_labels(labels)
    combined_labels = combined_labels.unsqueeze(1)
    decode_labels = torch.cat((labels, combined_labels), dim=1)
    decode_labels = decode_labels.repeat(1, 2, 1).view(-1, 3, 10)
    decode_labels[0, 0, :] = 0.
    decode_labels[1, 1, :] = 0.
    decode_labels[2, 0, :] = 0.
    decode_labels[3, 1, :] = 0.
    print(decode_labels)
    decoded = coder(decode_labels)
    print(decoded)