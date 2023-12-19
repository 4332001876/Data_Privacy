import random

import sympy


def is_primitive_root(g, p, factors):
    # determine whether g is a primitive root of p
    for factor in factors:
        if pow(g, (p-1)//factor, p) == 1:
            return False
    return True

def generate_p_and_g(n_bit):
    while True:
        # generate an n-bit random prime number p
        p = sympy.randprime(2**(n_bit-1), 2**n_bit)

        # compute the prime factorization of p-1
        factors = sympy.factorint(p-1).keys()

        # choose a possible primitive root g
        for g in range(2, p):
            if is_primitive_root(g, p, factors):
                return p, g
            

def mod_exp(base, exponent, modulus):
    """TODO: calculate (base^exponent) mod modulus. 
        Recommend to use the fast power algorithm.
    """
    result = None
    return result


def elgamal_key_generation(key_size):
    """Generate the keys based on the key_size.
    """
    # generate a large prime number p and a primitive root g
    p, g = generate_p_and_g(key_size)

    # TODO: generate x and y here.
    x = None
    y = None

    return (p, g, y), x

def elgamal_encrypt(public_key, plaintext):
    """TODO: encrypt the plaintext with the public key.
    """
    c1, c2 = None, None
    return c1, c2

def elgamal_decrypt(public_key, private_key, ciphertext):
    """TODO: decrypt the ciphertext with the public key and the private key.
    """
    plaintext = None
    return plaintext

if __name__ == "__main__":
    # set key_size, such as 256, 1024...
    key_size = int(input("Please input the key size: "))

    # generate keys
    public_key, private_key = elgamal_key_generation()
    print("Public Key:", public_key)
    print("Private Key:", private_key)

    # encrypt plaintext
    plaintext = int(input("Please input an integer: "))
    ciphertext = elgamal_encrypt(public_key, plaintext)
    print("Ciphertext:", ciphertext)

    # decrypt ciphertext
    decrypted_text = elgamal_decrypt(public_key, private_key, ciphertext)
    print("Decrypted Text:", decrypted_text)
