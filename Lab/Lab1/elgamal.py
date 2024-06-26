import random
import time

import sympy
import matplotlib.pyplot as plt


def is_primitive_root(g, p, factors):
    # determine whether g is a primitive root of p
    for factor in factors:
        if pow(g, (p-1)//factor, p) == 1:
            return False
    return True

def generate_p_and_g(n_bit):
    while True:
        # generate an n-bit random prime number p
        # 我们随机选取一个素数q，然后满足p=2q+1也是素数，这样p-1的质因数分解结果就是2*q
        finded = False
        while not finded:
            q = sympy.randprime(2**(n_bit-2), 2**(n_bit-1)) # 生成一个n_bit-1位的素数q，这可以保证p位数是n_bit
            p = 2 * q + 1
            finded = sympy.isprime(p)
            
        # compute the prime factorization of p-1
        factors = [2, q]

        # choose a possible primitive root g
        for g in range(2, p):
            if is_primitive_root(g, p, factors):
                return p, g
            

def mod_exp(base, exponent, modulus):
    """ calculate (base^exponent) mod modulus. 
        Recommend to use the fast power algorithm.
    """
    result = 1 # base^0 mod modulus = 1
    # 下面的操作是，从右到左取出exponent的每一位，如果该位为1，则将base^(2^i)乘（取模意义下）入到结果中
    # base^(2^i)以动态规划方法给出
    # 该算法使用了位运算优化，exponent & 1指取出exponent的最低位，exponent >>= 1指准备取exponent的下一位
    while exponent > 0:
        if exponent & 1:
            result = (result * base) % modulus
        base = (base * base) % modulus
        exponent >>= 1
    return result


def elgamal_key_generation(key_size):
    """Generate the keys based on the key_size.
    """
    # generate a large prime number p and a primitive root g
    p, g = generate_p_and_g(key_size)

    # *-TODO: generate x and y here.
    # 随机选择一个私钥x，1<=x<=p-2
    x = random.randint(1, p-2)
    # 计算公钥y
    y = mod_exp(g, x, p)

    return (p, g, y), x

def elgamal_encrypt(public_key, plaintext):
    """ encrypt the plaintext with the public key.
    """
    p, g, y = public_key
    # 随机选择一个临时密钥k，1<=k<=p-2
    k = random.randint(1, p-2)
    # 计算临时公钥c1
    c1 = mod_exp(g, k, p)
    # 计算密文c2
    c2 = (mod_exp(y, k, p) * plaintext) % p
    return c1, c2

def elgamal_decrypt(public_key, private_key, ciphertext):
    """ decrypt the ciphertext with the public key and the private key.
    """
    p, g, y = public_key
    c1, c2 = ciphertext
    # 利用私钥x计算临时公钥c1的模反演s
    s = mod_exp(c1, private_key, p)
    # 利用s计算明文
    s_inverted = sympy.mod_inverse(s, p) # 求s的逆元
    plaintext = (c2 * s_inverted) % p
    return plaintext

def main_interactive():
    # set key_size, such as 256, 1024...
    key_size = int(input("Please input the key size: "))

    # generate keys
    public_key, private_key = elgamal_key_generation(key_size)
    print("Public Key:", public_key)
    print("Private Key:", private_key)

    # encrypt plaintext
    plaintext = int(input("Please input an integer: "))
    ciphertext = elgamal_encrypt(public_key, plaintext)
    print("Ciphertext:", ciphertext)

    # decrypt ciphertext
    decrypted_text = elgamal_decrypt(public_key, private_key, ciphertext)
    print("Decrypted Text:", decrypted_text)


def main(key_size, plaintext):
    # generate keys
    public_key, private_key = elgamal_key_generation(key_size)
    # encrypt plaintext
    ciphertext = elgamal_encrypt(public_key, plaintext)
    # decrypt ciphertext
    decrypted_text = elgamal_decrypt(public_key, private_key, ciphertext)
    return decrypted_text

def main_encrypt_batch(key_size, plaintext_list):
    # generate keys
    public_key, private_key = elgamal_key_generation(key_size)
    ciphertext_list = []
    for plaintext in plaintext_list:
        # encrypt plaintext
        ciphertext = elgamal_encrypt(public_key, plaintext)
        ciphertext_list.append(ciphertext)
    decrypted_text_list = []
    for ciphertext in ciphertext_list:
        # decrypt ciphertext
        decrypted_text = elgamal_decrypt(public_key, private_key, ciphertext)
        decrypted_text_list.append(decrypted_text)
    return ciphertext_list, decrypted_text_list

def test_time():
    test_rounds = {
        64: 1000,
        128: 100,
        256: 10,
        512: 3,
        1024: 1
    }
    for key_size, rounds in test_rounds.items():
        print("Key size: ", key_size)
        # stage 1
        start = time.time()
        for i in range(rounds):
            public_key, private_key = elgamal_key_generation(key_size)
        end = time.time()
        print("Stage 1 time cost: ", (end-start) / rounds)
        # stage 2
        start = time.time()
        for i in range(rounds):
            plaintext = 123456789+i
            ciphertext = elgamal_encrypt(public_key, plaintext)
        end = time.time()
        print("Stage 2 time cost: ", (end-start) / rounds)
        # stage 3
        start = time.time()
        for i in range(rounds):
            decrypted_text = elgamal_decrypt(public_key, private_key, ciphertext)
        end = time.time()
        print("Stage 3 time cost: ", (end-start) / rounds)

def test_time_batch():
    test_rounds = {
        64: 1000,
        128: 1000,
        256: 1000,
    }
    for key_size, rounds in test_rounds.items():
        print("Key size: ", key_size)
        start = time.time()
        main_encrypt_batch(key_size, [123456]*rounds)
        end = time.time()
        print("new time cost: ", (end-start) / rounds)

        start = time.time()
        for i in range(rounds):
            main(key_size, 123456)
        end = time.time()
        print("origin time cost: ", (end-start) / rounds)

        
        

def test_randomness():
    key_size = 128
    ciphertext_list, _ = main_encrypt_batch(key_size, [123456]*100)
    for item in ciphertext_list:
        print(item)
    
def test_multiply_homomorphic(key_size):
    # generate keys
    public_key, private_key = elgamal_key_generation(key_size)
    plain_factor1 = 24
    plain_factor2 = 5
    # encrypt plaintext
    ciphertext1 = elgamal_encrypt(public_key, plain_factor1)
    ciphertext2 = elgamal_encrypt(public_key, plain_factor2)
    start = time.time()
    for _ in range(1000):
        factor1 = elgamal_decrypt(public_key, private_key, ciphertext1)
        factor2 = elgamal_decrypt(public_key, private_key, ciphertext2)
        test_product = factor1 * factor2
    end = time.time()
    print("decrypt([a])*decrypt([b]) time cost: ", (end-start) / 1000)

    start = time.time()
    for _ in range(1000):
        cipher_product = ((ciphertext1[0]*ciphertext2[0]) % public_key[0], (ciphertext1[1] * ciphertext2[1]) % public_key[0])
        product = elgamal_decrypt(public_key, private_key, cipher_product)
    end = time.time()
    print("decrypt([a]*[b]) time cost: ", (end-start) / 1000)

    print("decrypt([a]): ", factor1)
    print("decrypt([b]): ", factor2)
    print("decrypt([a]*[b]): ", product)
    print("decrypt([a])*decrypt([b]): ", factor1 * factor2)
    print("decrypt([a])*decrypt([b]) == decrypt([a]*[b])? ", factor1 * factor2 == product)

    # decrypt ciphertext
    decrypted_text = elgamal_decrypt(public_key, private_key, ciphertext1)
    return decrypted_text


if __name__ == "__main__":
    test_time_batch()
