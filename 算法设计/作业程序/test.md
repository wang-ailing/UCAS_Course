### 根据题中的一般步骤，证明**精确覆盖问题**和**三元精确覆盖问题**是**NPC问题**：

---

### **1. 精确覆盖问题（Exact Cover Problem）的证明**

**问题描述**：  
给定一个集合 \( X \) 和一个集合族 \( S = \{ S_1, S_2, \dots, S_m \} \)，每个 \( S_i \subseteq X \)。是否存在 \( S \) 的子集 \( S' \subseteq S \)，使得 \( S' \) 中的集合两两不相交，并且它们的并集等于 \( X \)。

**证明步骤**：

#### (a) 证明 \( \Pi \in NP \)：  
- 对于一个给定解 \( S' \subseteq S \)，我们可以在**多项式时间**内验证：
  1. \( S' \) 中的集合是否两两不相交；
  2. \( S' \) 中的所有集合的并集是否等于 \( X \)。
- 因此，精确覆盖问题属于 NP 类。

#### (b) 选择一个已知的 NP完全问题 \( \Pi' \)：  
- 选择布尔可满足性问题（SAT）或**集合覆盖问题（Set Cover）**作为已知的 NP完全问题。

#### (c) 构造一个从 \( \Pi' \) 到 \( \Pi \) 的转换 \( f \)：  
- 将集合覆盖问题（Set Cover）转换为精确覆盖问题：  
  1. 对于集合覆盖问题的每个元素 \( e_i \)，我们将其表示为 \( X \) 中的元素。  
  2. 将集合覆盖问题的集合族 \( S \) 直接对应到精确覆盖问题中的集合族 \( S \)。  
  3. 修改约束，要求所有集合在精确覆盖问题中**互不重叠**，并且完全覆盖 \( X \)。  
- 这种转换在多项式时间内完成。

#### (d) 证明 \( f \) 是一个多项式变换：  
- 从集合覆盖问题到精确覆盖问题的转换只涉及简单的集合构建和条件修改，时间复杂度是多项式的。

**结论**：  
- 精确覆盖问题满足：属于 NP，且可以通过已知的 NP完全问题多项式归约得到。  
- 因此，精确覆盖问题是 NP完全问题。

---

### **2. 三元精确覆盖问题（Exact 3-Cover, X3C）的证明**

**问题描述**：  
给定一个集合 \( X \) 和一个集合族 \( S = \{ S_1, S_2, \dots, S_m \} \)，其中 \( |S_i| = 3 \)（每个集合包含3个元素）。是否存在 \( S' \subseteq S \)，使得 \( S' \) 中的集合两两不重叠，并且它们的并集等于 \( X \)？

**证明步骤**：

#### (a) 证明 \( \Pi \in NP \)：  
- 对于给定的解 \( S' \subseteq S \)，我们可以在**多项式时间**内验证：
  1. \( S' \) 中的集合是否两两不相交；
  2. \( S' \) 中的所有集合的并集是否等于 \( X \)。  
- 因此，三元精确覆盖问题属于 NP 类。

#### (b) 选择一个已知的 NP完全问题 \( \Pi' \)：  
- 选择精确覆盖问题（Exact Cover）作为已知的 NP完全问题。

#### (c) 构造一个从 \( \Pi' \) 到 \( \Pi \) 的转换 \( f \)：  
- 将精确覆盖问题中的任意大小的集合族 \( S \) 转换为三元精确覆盖问题：  
  1. 如果 \( |S_i| > 3 \)：将 \( S_i \) 拆分成多个子集，使得每个子集大小为3，且这些子集之间共享元素；  
  2. 如果 \( |S_i| < 3 \)：添加虚拟元素（填充元素）将其扩展到大小为3。  
- 转换后的问题依然满足覆盖 \( X \) 的要求，且所有集合大小均为3。

#### (d) 证明 \( f \) 是一个多项式变换：  
- 这个转换过程涉及集合拆分和元素填充，时间复杂度是多项式的。

**结论**：  
- 三元精确覆盖问题满足：属于 NP，且可以通过已知的 NP完全问题（精确覆盖问题）多项式归约得到。  
- 因此，三元精确覆盖问题是 NP完全问题。

---

### 总结  
根据上述证明步骤，**精确覆盖问题** 和 **三元精确覆盖问题** 都属于 **NP完全问题**，因为它们满足：
1. 属于 NP 类（解可以在多项式时间内验证）。  
2. 通过已知的 NP完全问题（如集合覆盖问题或精确覆盖问题）进行多项式时间归约。

这两个问题在计算复杂性理论中都是重要的经典 **NPC问题**。