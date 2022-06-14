import torch
import random
import warnings
# num bits      sign    exponent    mantissa
# 64 bit:       1       11          52
# 32 bit:       1       8           23
# 16 bit:       1       5           10

# from: https://github.com/KarenUllrich/pytorch-binary-converter/blob/master/binary_converter.py
def bit2float(b, num_e_bits=8, num_m_bits=23, bias=127.):
    """Turn input tensor into float.
        Args:
            b : binary tensor. The last dimension of this tensor should be the
            the one the binary is at.
            num_e_bits : Number of exponent bits. Default: 8.
            num_m_bits : Number of mantissa bits. Default: 23.
            bias : Exponent bias/ zero offset. Default: 127.
        Returns:
            Tensor: Float tensor. Reduces last dimension.
    """
    expected_last_dim = num_m_bits + num_e_bits + 1
    assert b.shape[-1] == expected_last_dim, "Binary tensors last dimension " \
                                           "should be {}, not {}.".format(
                                            expected_last_dim, b.shape[-1])

    # check if we got the right type
    dtype = torch.float32
    if expected_last_dim > 32: dtype = torch.float64
    if expected_last_dim > 64:
        warnings.warn("pytorch can not process floats larger than 64 bits, keep"
                  " this in mind. Your result will be not exact.")

    s = torch.index_select(b, -1, torch.arange(0, 1))
    e = torch.index_select(b, -1, torch.arange(1, 1 + num_e_bits))
    m = torch.index_select(b, -1, torch.arange(1 + num_e_bits,
                                             1 + num_e_bits + num_m_bits))
    # SIGN BIT
    out = ((-1) ** s).squeeze(-1).type(dtype)
    # EXPONENT BIT
    exponents = -torch.arange(-(num_e_bits - 1.), 1.)
    exponents = exponents.repeat(b.shape[:-1] + (1,))
    e_decimal = torch.sum(e * 2 ** exponents, dim=-1) - bias
    out *= 2 ** e_decimal
    # MANTISSA
    matissa = (torch.Tensor([2.]) ** (
    -torch.arange(1., num_m_bits + 1.))).repeat(
    m.shape[:-1] + (1,))
    out *= 1. + torch.sum(m * matissa, dim=-1)
    return out


def float2bit(f, num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32):
    """Turn input tensor into binary.
        Args:
            f : float tensor.
            num_e_bits : Number of exponent bits. Default: 8.
            num_m_bits : Number of mantissa bits. Default: 23.
            bias : Exponent bias/ zero offset. Default: 127.
            dtype : This is the actual type of the tensor that is going to be
            returned. Default: torch.float32.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.
    """
    ## SIGN BIT
    s = torch.sign(f)
    f = f * s
    # turn sign into sign-bit
    s = (s * (-1) + 1.) * 0.5
    s = s.unsqueeze(-1)

    ## EXPONENT BIT
    if torch.eq(f, 0.).item():
        return torch.zeros(32)
    else:
        e_scientific = torch.floor(torch.log2(f))
    e_decimal = e_scientific + bias
    e = integer2bit(e_decimal, num_bits=num_e_bits)

    ## MANTISSA
    m1 = integer2bit(f - f % 1, num_bits=num_e_bits)
    m2 = remainder2bit(f % 1, num_bits=bias)
    m = torch.cat([m1, m2], dim=-1)

    dtype = f.type()
    idx = torch.arange(num_m_bits).unsqueeze(0).type(dtype) + (8. - e_scientific).unsqueeze(-1)
    idx = idx.long()
    idx = torch.squeeze(idx)
    m = torch.gather(m, dim=-1, index=idx)

    return torch.cat([s, e, m], dim=-1).type(dtype)


def remainder2bit(remainder, num_bits=127):
    """Turn a tensor with remainders (floats < 1) to mantissa bits.
        Args:
            remainder : torch.Tensor, tensor with remainders
            num_bits : Number of bits to specify the precision. Default: 127.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.
    """
    dtype = remainder.type()
    exponent_bits = torch.arange(num_bits).type(dtype)
    exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
    out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
    return torch.floor(2 * out)


def integer2bit(integer, num_bits=8):
    """Turn integer tensor to binary representation.
        Args:
            integer : torch.Tensor, tensor with integers
            num_bits : Number of bits to specify the precision. Default: 8.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.
    """
    dtype = integer.type()
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    out = integer.unsqueeze(-1) / 2 ** exponent_bits
    return (out - (out % 1)) % 2

def flip_bit(tensor, ind:int=-1, num_bits:int=32, ind_range:tuple=None):
    # set values for number of bits in e, m, b
    if num_bits == 32:
        e = 8
        m = 23
        b = 127.
    elif num_bits == 64:
        e = 11
        m = 52
        b = 1023.
    elif num_bits == 16:
        e = 5
        m = 10
        b = 15.
    else:
        raise ValueError
    
    # convert float to bit representation
    bit_float = float2bit(tensor, num_e_bits=e, num_m_bits=m, bias=b)
    
    # TODO: provide option for ind range
    # if no ind given, use a random
    if ind == -1:
        if ind_range is not None:
            ind = random.randint(0, bit_float.shape[0] - 1)
        else:
            ind = random.randint(ind_range)
        
    # flip the bit
    bit = bit_float[ind].item()
    if bit == 0:
        flip = 1.
    else:
        flip = 0.
    
    # return the float representation
    bit_float[ind] = flip
    return bit2float(bit_float, num_e_bits=e, num_m_bits=m, bias=b)   