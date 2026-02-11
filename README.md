# ClaimTaxo
A claim-centered, bidirectional taxonomy induction framework.

## Conversation history with ChatGPT

### Me:

我重新明确一下我们的方法。我们先过滤posts得到更有可能包含claim的posts。下一步，我们根据全局语料构建初始taxonomy（如同topics-subtopics-claim-aspects-argument这样的结构，不严格要求，包含concept memory bank），然后把语料按时间线切分，再从前往后更新taxonomy。具体来说，在每一个slice内，我们先把和已有argument节点（我们可以通过claim-aspects-argument来构造语句）的embeddings最匹配的posts归于argument节点。对于unmapped pool进行聚类和采样，然后根据它们来更新taxonomy（不能删除老的）。处理unmapped pool的过程应该是迭代直到满足条件的对吧。注意，每个slice每一个post的指派都需要被记录下来。同时，需要跨时间维护一个重定向表。这样，在最新的taxonomy构建完成后，过去所有有过指派的post都能对应到最新的taxonomy上，然后我们把这个taxonomy放到各个slices上，就得到了consistent的每个slice的taxonomy。下一步，我们抽样每个slice里argument节点对应的post，就能得到stance的分布。
这个过程对吗？处理unmapped pool的过程应该和taxoadapt很像，要不横向增长要不纵向深入。不过终止条件可能比较难定义。

### GPT:

整体流程对的，而且你把“版本化 taxonomy + 不删除 + 重定向表 + 事后统一口径”的关键点都说清楚了。你现在卡住的其实是两件事：
	1.	unmapped pool 的迭代扩展怎么做才不陷入无穷/过拟合
	2.	终止条件怎么定义才像 research，而不是拍脑袋阈值

我下面给你一套“像 TaxoAdapt 的 width/depth 扩展，但更适配社媒 + concept memory bank”的 unmapped 处理与终止机制。重点是：终止条件用“覆盖率的边际收益”+“一致性风险”来定，而不是单一超参。

⸻

你描述的流程哪里需要微调？

✅ 需要补的一个结构：把“指派记录”分两层存

你说“每个 slice 每个 post 的指派都要记录”，对；但要避免后面 split/merge 造成难维护，我建议记录成两层：
	•	Raw assignment：post → node_id_at_time_t（当时的节点ID）
	•	Canonical assignment：post → canonical_node_id（通过 redirect 表在线/离线折叠出来）

这样你在任何时刻都能：
	•	复现当时系统怎么看（审计）
	•	同时得到统一口径的统计（canonical）

⸻

unmapped pool 的处理确实应该迭代，但迭代的“单位”要换

不是“反复让 LLM 看 unmapped”，而是：

每次迭代只处理 top-K 个最值得结构化的 unmapped cluster，其余留到后面或忽略。

否则社媒里噪声会把你拖死。

⸻

终止条件：用两个“边际收益曲线”来收敛（核心）

在每个 slice，你的目标不是“把所有东西都塞进 taxonomy”，而是：
	•	覆盖主要讨论
	•	不因为噪声产生大量一次性节点
	•	保持跨时间一致

所以终止应该由收益递减驱动，而不是“unmapped < 某阈值”这种不稳的条件。

我给你两条同时满足的终止准则：

⸻

终止准则 1：Coverage Gain Saturation（覆盖率增益饱和）

定义本 slice 的“可解释覆盖率”：
	•	mapped 到已有节点的 posts 占比
	•		•	新增节点后被吸收的 posts 占比

每做一次“unmapped→聚类→采样→LLM 决策→新增/挂接/拆分”的迭代，你都能算一个增益：

\Delta \text{Coverage} = \text{Coverage}_{k} - \text{Coverage}_{k-1}

当连续 R 次迭代，ΔCoverage 小于 ε，就停止。

这比绝对阈值稳，因为它不依赖你提前知道“unmapped 应该剩多少”。

工程上：
	•	ε 可以是 0.5% 或 1% 级别
	•	R=2 或 3（避免偶然）

这条非常“research”：它是边际收益收敛。

⸻

终止准则 2：One-off Risk Control（一次性节点风险控制）

社媒最容易出的问题是：你为一堆噪声/梗图/转述生成了很多新节点，这会破坏一致性。

所以你需要一个“新节点是否值得进入全局口径”的门槛，但别用固定频次阈值（你自己也不喜欢）。

更稳的做法是：看它是不是“只在当前 slice 存在”。

你可以给每个新节点一个“试用期”状态：
	•	candidate：刚发现
	•	promoted：跨多个 slice 有持续证据
	•	alias/deprecated：证明是一次性噪声

终止时，如果本 slice 新增的绝大多数节点都只能停留在 candidate（缺乏足够权重/语义稳定性），说明你开始在挖噪声：

当新增节点中 candidate 比例过高且覆盖增益又很小，就停止。

这条本质上是：防止结构无限膨胀。

⸻

unmapped 的迭代策略：像 TaxoAdapt，但信号更适配你

你说得对：会像 TaxoAdapt 的 width/depth。区别是你不用它那种 heavy LLM 分类，而是靠“密度/未解释密度 + 语义新颖性”来触发。

在 slice 内对父节点 p：
	•	Width expansion（横向）信号：
有一群 posts 总被映射到 p（或 p 的某个粗节点），但对任何已有子节点都低匹配 → 说明子类不全
	•	Depth expansion（纵向）信号：
某个叶节点的 evidence 很大且内部出现稳定子簇 → 说明需要细分

这跟 TaxoAdapt 的“density / unmapped density”思想一致，只是你把对象换成社媒讨论与 argument 节点。它用密度信号决定扩展方向的出发点你完全可以引用借鉴。 ￼

⸻

迭代“处理哪些 cluster”才不会 trivial？

给 unmapped 聚类后，不是全处理。你每轮只处理最值得的，排序可以用一个很简单、但非常合理的优先级：

Priority = 规模 × 新颖性 × 时间集中度
	•	规模：cluster 覆盖多少条
	•	新颖性：离最近已有节点有多远（embedding distance）
	•	时间集中度：是不是在短窗口内爆发（你之前的 burst 想法，这里可以作为排序信号，而不是单独模块）

这会自动实现你想要的“局部密集不能漏”，而不用额外塞一个大模块。

⸻

你最后那句：抽样 argument 节点的 posts 得 stance 分布 ——这里要小心

如果你只抽样然后用 LLM 判 stance，你得到的是估计分布，不是精确分布。

但这完全可以是你的研究贡献（而且比“逐条 LLM”更合理）。关键是：
	•	抽样要分层：按时间、按 cluster、按不确定性
	•	输出要带置信区间：这会让审稿人觉得你很严谨

你甚至可以在论文里明确：

We estimate stance distributions via stratified sampling with evidence-backed uncertainty bounds, avoiding full LLM enumeration.

这一下就不 trivial 了。

⸻

小结：你的流程对，但终止条件建议这样写（可论文化）

在每个 slice 内，unmapped 处理迭代直到同时满足：
	1.	Coverage Gain 饱和：连续 R 轮 ΔCoverage < ε
	2.	结构膨胀受控：新增节点多为 candidate 且增益很小（one-off risk）
	3.	（可选）达到预算：本 slice 最多允许 LLM 决策 M 个 cluster（现实约束）

这三个一起，终止就很自然，而且不会被喷“阈值拍脑袋”。

⸻

如果你愿意，我下一步可以把你整套方法写成一个清晰的 Algorithm 1 伪代码（含：slice 循环、映射、unmapped 迭代、edit-op、redirect 更新、assignment 记录、终止判断、以及第二阶段 stance 抽样流程）。这样你直接就有 paper 的方法部分骨架了。

### Me:

你把完整的框架流程复述一遍

### GPT:

下面是你现在这套完整框架流程的“从输入到输出”的复述版（按你定义的两阶段 + 时间增量维护 + concept memory bank + redirect 表）：

⸻

0. 输入与目标

输入：一批按时间戳排序的 social media posts。
输出：
	1.	一棵（或多层）跨时间一致的 taxonomy（例如 topics → subtopics → claims → aspects → arguments），每个节点都有 Concept Memory Bank（CMB）。
	2.	对每个时间 slice，都能得到同一口径（canonical）的 taxonomy 视图及节点统计。
	3.	在最终 taxonomy 上，对每个 slice 的 argument 节点估计 stance 分布。

⸻

Phase 0：Claim-like posts 过滤
	1.	对全量 posts 做过滤，保留更可能包含 claim / check-worthy 内容的 posts（减少噪声与成本）。
	2.	得到过滤后的语料库 D。

⸻

Phase 1：构建初始 taxonomy + Concept Memory Bank
	3.	用全局语料 D 做 clustering + 采样，构建一个初始的多层 taxonomy（不要求严格固定形态，但要包含 argument 层）。
	4.	为 taxonomy 的每个节点建立 CMB（定义/边界/关键词/代表例子/父子关系等）。
	5.	初始化两个关键记录结构：

	•	Assignment log：记录 post 在每个 slice 的节点指派（post_id, slice_id, node_id_at_time）。
	•	Redirect table（重定向表）：记录节点的 merge/split/rename 等演化关系，用于把旧节点折叠到新节点（canonicalization）。
（并约束：节点不能删除，只能 inactive / redirect / superseded。）

⸻

Phase 2：按时间 slice 增量更新 taxonomy（从前往后遍历）
	6.	将语料按时间线切分为 slices：D = \{D_1, D_2, ..., D_T\}。
	7.	从 t=1 到 T 依次处理每个 slice。

⸻

2.1 Slice 内：先映射到已有 argument 节点
	8.	对当前 slice D_t，把每条 post 和当前 taxonomy 的 argument 叶节点做 embedding 匹配：

	•	匹配得上的 posts → 归入对应 argument 节点的 evidence pool
	•	匹配不可靠/匹配不上 → 放入 unmapped pool

	9.	对所有已归入节点的 posts，写入 Assignment log：

	•	记录 (post_id, t, node_id_current)

⸻

2.2 Slice 内：迭代处理 unmapped pool，触发 taxonomy 更新（不能删旧节点）
	10.	对 unmapped pool 做聚类（目的是把明显的新讨论聚成若干簇），并对每簇采样代表帖形成 evidence pack。
	11.	基于这些 evidence packs，结合当前 taxonomy + CMB，让系统决定对 taxonomy 做哪类更新（类似 TaxoAdapt 的 width/depth 思路）：

	•	横向增长（width）：补齐同层新的 argument 节点 / 新 sibling
	•	纵向深入（depth）：对某个已有节点 split 出更细的子 argument
	•	merge/rename/refine：合并同义、改名、补边界（但不删除旧节点）

	12.	每次发生结构变更时：

	•	写入 Redirect table（比如 old→new、superseded→children）
	•	更新相关节点的 CMB（定义/边界/例子/关键词等）

	13.	taxonomy 更新后，对 slice 中受影响的 posts 重新映射（至少对 unmapped pool 及变更相关节点的邻域重跑映射）。
	14.	重复步骤 10–13（即：unmapped→聚类采样→更新 taxonomy→重新映射），直到满足 slice 内终止条件（比如覆盖增益饱和/预算用完/新增节点边际收益很小等）。
	15.	在 slice 结束时，你得到：

	•	更新后的 taxonomy（版本 t）
	•	完整的 slice 内 post→节点指派记录（Assignment log）
	•	累积更新过的 Redirect table

⸻

Phase 3：遍历完所有 slices 后做“口径统一”
	16.	遍历结束得到最终版本 taxonomy（最新的 canonical taxonomy）+ 最终的 Redirect table。
	17.	用 Redirect table 对 Assignment log 做 canonicalization：

	•	把历史上 (post_id, t, node_id_at_time) 折叠成 (post_id, t, canonical_node_id_final)

这样保证：过去所有 slice 的指派都能映射到最终 taxonomy 的节点ID上。
	18.	将最终 taxonomy 投影到每个 slice：

	•	每个 slice 都使用同一套节点 ID / 结构口径
	•	slice 只在节点的“活跃度/证据/统计”上不同
→ 得到 consistent 的每-slice taxonomy。

⸻

Phase 4：stance 分布估计（第二阶段）
	19.	对每个 slice、每个 canonical argument 节点，从该节点在该 slice 的 posts 中做抽样（可分层抽样）。
	20.	用 LLM 对抽样的 posts（相对于该 argument 节点）判定 stance（support/oppose/neutral 等）。
	21.	用抽样结果估计该 (slice, argument node) 的 stance 分布，并汇总成时间序列（每个 argument 节点一条 stance 曲线）。

⸻

如果你愿意，我也可以把上面流程压成论文里常见的 6–8 行 “Pipeline overview”版本，或者写成 Algorithm 1 伪代码格式（含关键数据结构：CMB、assignment log、redirect table、slice-level iteration）。

## 确保API key已设置
export OPENAI_API_KEY='sk-5aa5ea0263c942f896210d035529b47b'

## 运行
bash run_naloxone_taxonomy.sh