## 6
定义集合$S = \{i|i \in [n], \epsilon_i \ge t\}, T=\{i|i \in [n], \epsilon_i < t\}$

由此，对于两个距离为d的数据集$D_{0}, D_{d}$，可构造一连串的数据集$D_{0},D_{1}, D_{2}, ..., D_{(d-1)}, D_{d}$，每两个在序列中相邻的数据集之间满足距离为1，则有：
$$
\frac{P(M(D_{d}) \in S)}{P(M(D_{0}) \in S)} = \prod_{i=0}^{d-1} \frac{P(M(D_{i+1}) \in S)}{P(M(D_{i}) \in S)} \le \prod_{i=0}^{d-1} e^t = e^{td}
$$