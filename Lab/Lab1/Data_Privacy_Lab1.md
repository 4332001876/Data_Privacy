# Data_Privacy_Lab1
**罗浩铭 PB21030838**

## DP-SGD

### 代码实现
#### 实验框架debug
代码中的梯度下降公式有误，应改为如下形式（第一条公式可选择下面CrossEntropyLoss与L2Loss中的一种）：
```python
dz = -(y / (predictions + self.tau) - (1 - y) / (1 - predictions + self.tau)) # Cross entropy loss
# dz = predictions - y # L2_loss
dz = dz * (predictions * (1 - predictions)) # sigmoid derivative

dw = np.dot(X.T, dz) / num_samples
db = np.sum(dz) / num_samples
```

同时数据集的输入没有归一化，各输入特征的取值范围差异巨大（至少差4个数量级），导致训练过程中的动力学特性非常差，因此需要加入归一化。
```python
# normalize the data
X = (X - np.mean(X, axis=0)) / X.std(axis=0)
```

#### 数据集背景介绍
本次实验运行在数据集breast cancer dataset上，该数据集包含569个样本，每个样本有30个特征，表示该肿瘤各特征，每个样本都有一个标签，标签为0或1，表示该样本是否为恶性肿瘤。训练时，将其中455个样本作为训练集，114个样本作为测试集。

本次实验的目的是使用DP-SGD算法训练一个二分类器，使得该二分类器能够对新的样本进行预测，判断该样本是否为恶性肿瘤。

#### 使用Advanced Composition Theorem计算每个epoch所需隐私保证
我们使用以下方式计算在总的隐私保证要求为$\epsilon, \delta$时，每个epoch需要的隐私保证：
取$\delta = \delta'$，则计算每个epoch的隐私保证为$\epsilon_u, \delta_u$如下：
先计算$\delta_u$:
$$
\delta_u = \frac{\delta}{k + 1}
$$
由于Advanced Composition Theorem中关于$\epsilon_u$的计算公式如下：
$$
\epsilon_u = \sqrt{2k \ln(1/\delta')} \epsilon + k \epsilon (e^{\epsilon} - 1) 
$$
则由$\epsilon_u$解出$\epsilon$需要求解超越方程，我们只能对其进行近似计算：
第一步，省略方程右半部分，近似计算$\epsilon_u$的上界：
$$
\epsilon_1 = \epsilon_u / \sqrt{2k \ln(1/\delta')} 
$$
第二步，将第一步计算出的$\epsilon_1$代入方程右半部分，进行修正，由于代入的$\epsilon_1$偏大，计算所得$\epsilon_2$偏小，这将其不超过隐私预算限制：

$$
\epsilon_2 = \epsilon_u / (\sqrt{2k \ln(1/\delta')} + k (e^{\epsilon_1} - 1) )
$$

最终以$\epsilon_2$作为每个epoch的隐私保证，实测在100-10000这一epoch数范围内，该估算使得最后花费的总隐私预算与隐私预算上限的差异不超过0.2%。

```python
# 由Advanced Composition Theorem算出每个epoch需要的隐私保证
k = self.num_iterations
# epsilon_u, delta_u = epsilon / k, delta / k
delta_u = delta / (k + 1)
# 近似计算epsilon_u
# 第一步，近似计算epsilon_u，这一步的计算结果会偏大
epsilon_u = epsilon / np.sqrt(2 * k * np.log(1 / delta_u))
# 第二步，进行修正，分母中的epsilon_u大于真实的epsilon_u，因此可以保证epsilon_u的计算结果偏小，符合隐私保证
epsilon_u = epsilon / (np.sqrt(2 * k * np.log(1 / delta_u)) + k * (np.exp(epsilon_u) - 1))
```

#### 梯度裁剪
梯度裁剪用于保证每个样本数据的敏感度被限制在C以内。
这里的敏感度即为梯度的L2范数，因此我们只需要计算每个样本的梯度的L2范数，然后将其与C比较，如果大于C，则将其缩放为C，否则不变。

将以上思路向量化之后为如下代码，此处实现的是论文的原版算法：

```python
def clip_gradients(sample_gradients, C):
    # *-TODO: Clip gradients.
    grad_norm = np.linalg.norm(sample_gradients, ord=2, axis=1)
    clip_factor = np.maximum(1, grad_norm/C)
    clip_gradients = sample_gradients / clip_factor[:, np.newaxis]
    return clip_gradients
```

这里，传入的`sample_gradients`需要含有每一个样本的梯度，因此原有公式计算的梯度均值不适用，需把梯度在样本维度展开，重新计算为：`dw = X * dz.reshape(-1, 1)`。该梯度被作为函数的输入参数。

此后，计算每个样本的梯度的L2范数；然后，用`np.maximum(1, grad_norm/C)`来将确定是否缩放梯度与缩放系数大小的过程向量化，若`grad_norm`比C大，则缩放系数为`grad_norm/C`，使得缩放后梯度范数为C，否则无需缩放，缩放系数为1；最后将梯度与缩放系数相除（这里numpy的广播机制保证除法可以正确进行），得到裁剪后的梯度。

#### 添加高斯噪声
此处实现的是论文的原版算法：
```python
def add_gaussian_noise_to_gradients(sample_gradients, epsilon, delta, C):
    # *-TODO: add gaussian noise to gradients.
    gradients = np.sum(sample_gradients, axis=0)

    std = C * np.sqrt(2 * np.log(1.25 / delta)) / epsilon # 上述梯度裁剪已经保证了C即为sensitivity的上界
    noise = np.random.normal(loc=0.0, scale=std, size=gradients.shape)
    noisy_gradients = gradients + noise

    noisy_gradients = noisy_gradients / sample_gradients.shape[0]

    return noisy_gradients
```

首先，将上一步得到的每个样本的梯度求和，得到总的梯度。
然后以`epsilon, delta`以及敏感度`C`计算高斯噪声的标准差，生成高斯噪声并加到总的梯度上。
最后除以样本数量（相当于转成梯度均值），得到最终的梯度。


### 实验结果
baseline的参数如下：

|      参数      |  值   |
| :------------: | :---: |
| num_iterations | 1000  |
|    epsilon     |  1.0  |
|     delta      | 1e-3  |
| learning_rate  | 0.01  |
|  random_seed   |   1   |

以下每组实验将在baseline的基础仅修改一个来进行。

#### 不同差分隐私预算对于模型效果的影响
本组实验保证其它参数不变，且每个实验仅修改一个参数。



#### 不同的迭代轮数对于模型效果的影响
本组实验保证其它参数不变，也即保证相同的总隐私消耗量。


## Elgamal

### 代码实现
#### 快速幂工具函数
由于Elgamal算法中需要进行大数的取模意义下的快速幂运算，因此我们需要实现一个快速幂工具函数，用于计算大数在取模意义下的快速幂。

```python
def mod_exp(base, exponent, modulus):
    """*-TODO: calculate (base^exponent) mod modulus. 
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
```

该算法的基本思想是：初始化result为`base^0 mod modulus = 1`，此后从右到左取出exponent的每一位，如果该位为1，则将base^(2^i)乘（取模意义下）入到result中。

算法进行了优化：base^(2^i)以动态规划方法给出，每一轮只需以上一次的结果base^(2^(i-1))在取模意义下计算平方即可；该算法使用了位运算优化，exponent & 1指取出exponent的最低位，exponent >>= 1指准备取exponent的下一位，这将使程序的运行加快。

#### 第一阶段：密钥生成
首先调用函数`generate_p_and_g`生成大素数p及其原根g，然后调用函数`generate_keys`生成公钥和私钥。此处对`generate_p_and_g`函数进行了优化，使之避免了时间开销极端巨大的质因数分解操作，这将在实验报告后面优化部分详述。
然后，使用`random.randint`随机生成一个1至p-2的数作为私钥x，以此计算公钥$y = g^x \ mod \ p$，最后返回公钥$(p, g, y)$和私钥$x$。

实现如下：
```python
def elgamal_key_generation(key_size):
    """Generate the keys based on the key_size.
    """
    # generate a large prime number p and a primitive root g
    p, g = generate_p_and_g(key_size)

    # 随机选择一个私钥x，1<=x<=p-2
    x = random.randint(1, p-2)
    # 计算公钥y
    y = mod_exp(g, x, p)

    return (p, g, y), x
```

#### 第二阶段：加密
根据Elgamal算法，
此处使用`random.randint`随机生成一个1至p-2的数作为临时私钥k，以此计算临时公钥$c_1 = g^k \ mod \ p$，以及密文$c_2 = m \cdot y^k \ mod \ p$，其中m为明文。

实现如下：
```python
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
```


#### 第三阶段：解密
根据Elgamal算法，解密方式为：利用私钥x计算临时公钥$c_1$的模反演$s = c_1^x \ mod \ p$，然后利用$s$计算明文$m$，由$c_2 \cdot s^{-1} \ mod \ p$得到明文。
模逆元的计算使用了sympy库中的`sympy.mod_inverse`函数。

实现如下：
```python
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
```


### 测试三阶段时间开销


### 验证 ElGamal 算法的随机性以及乘法同态性质

#### 验证 ElGamal 算法的随机性


#### 验证 ElGamal 算法的乘法同态性质
对比乘法同态性质运算的时间开销，即 time(decrypt([a]*[b])) 和
time(decrypt([a])*decrypt([b]))，并给出原因说明。





### 优化 ElGamal 算法加解密的时间开销




