# Primitive Recovery 技术总结

## 1. 工作目标

当前 primitive recovery 部分的目标不是直接完成最终机器人抓取，而是先构建一个**面向透明玻璃器皿的结构化几何恢复原型**，验证以下问题是否可行：

1. 是否可以只依赖 `mask + depth(+ camera intrinsics)` 对简单玻璃器皿进行参数化拟合；
2. 是否可以把器皿从“无结构点云/深度图”转换为“可解释的 primitive 几何表示”；
3. 这种结构化表示是否适合作为后续 affordance surface 和 grasp proposal 的基础。

当前实现的 primitive recovery 是一个 **MVP（最小可运行版本）**，重点是先把完整链路跑通，而不是在现阶段追求高精度拟合。

---

## 2. 总体算法逻辑

当前 primitive recovery 的基本流程如下：

```text
RGB / depth / depth_completed / object mask
    ↓
2D 几何特征提取
    ↓
primitive 参数初始化
    ↓
3D primitive 渲染（mask + depth）
    ↓
多项 loss 约束下的参数优化
    ↓
输出结构化参数和可视化结果
```

这里的核心思想是：

- 不直接把透明器皿当成普通点云或深度图来处理；
- 而是利用玻璃器皿“几何规则、可参数化”的特点；
- 用一个低维、可解释的 primitive 表示去解释观测。

---

## 3. 目前采用的 primitive 建模方式

当前已经实现的对象是 **beaker（烧杯）**。

最开始的想法是把 beaker 近似成一个简单圆柱体，然后根据观测 mask 和 depth 优化以下参数：

- shape parameters:
  - `radius`
  - `height`
- pose parameters:
  - `x, y, z`
  - `roll, pitch, yaw`

即：

\[
\theta = (x, y, z, roll, pitch, yaw, radius, height)
\]

其中：

- `x, y, z` 是烧杯在相机坐标系中的三维位置
- `roll, pitch, yaw` 是烧杯姿态
- `radius, height` 是几何尺寸

这比最早的二维近似版本更完整，因为它能够表达真正的 3D 位置和姿态，而不是只在图像平面里移动一个二维形状。

---

## 4. 为什么先从 beaker 开始

当前先选 beaker 作为 primitive fitting 的对象，是因为它相比其他玻璃器皿最简单：

- 形状规则
- 近似圆柱
- 没有明显复杂 neck
- 便于先验证整个 pipeline

也就是说，beaker 是最适合作为 primitive recovery MVP 的第一类器皿。

---

## 5. 输入信息与前提

当前拟合依赖以下输入：

1. `mask`
   - 目标物体在图像上的前景区域
   - 当前来自 FastSAM 分割结果，后续也可以替换成人工标注或更高质量的实例 mask

2. `depth` / `depth_completed`
   - 当前主要使用 `depth_completed`
   - 它比 raw depth 更平滑、缺失更少，但不是严格的几何真值

3. `camera intrinsics`
   - 主要是 `fx, fy, cx, cy`
   - 当前 RealSense 数据是 **depth 对齐到 RGB**，因此应该使用 **color intrinsics**

4. 可选 `rgb`
   - 用于可视化或交互式选框

---

## 6. 目前的初始化方法

在真正开始优化之前，当前 pipeline 先从 mask 和 depth 中提取一些几何量，用来初始化 primitive 参数。

### 6.1 2D 几何特征

从 mask 中可以直接提取：

- `bbox`
- `mask centroid`
- `rotated bounding box`
- `principal axis`
- `top width`
- `width profile`

这些量主要用于：

- 给 primitive 一个合理的初始尺度
- 给 primitive 一个大致的方向
- 避免优化从完全随机参数开始

### 6.2 初始三维位置

当前方法是：

1. 在 mask 上取中心点 `(u, v)`
2. 在 mask 内取一个中位深度 `z`
3. 用相机内参把 `(u, v, z)` 反投影到相机坐标系，得到 `(x, y, z)`

这一步把 primitive 初始化从“图像平面位置”升级到了“真正的 3D 位置”。

### 6.3 初始尺寸

当前方法是：

- 从 mask 的短边宽度近似估计 `radius`
- 从 mask 的长边近似估计 `height`

这一步比较粗糙，但足够作为 first-stage initialization。

---

## 7. 当前渲染方式

当前 beaker 的渲染方式已经从最早的“二维矩形 + 椭圆”升级成了更接近 3D 的版本。

### 7.1 当前做法

1. 在局部坐标系中采样一个圆柱体表面点集
2. 通过 `roll, pitch, yaw` 做旋转
3. 再加上 `(x, y, z)` 做平移
4. 用相机内参把 3D 点投影到图像上
5. 根据投影结果生成：
   - `rendered_mask`
   - `rendered_depth`

### 7.2 当前意义

这一步的意义是：

- primitive 已经不再是纯二维图形
- 而是真正在相机坐标系中被解释成一个 3D 物体

---

## 8. 当前优化目标

当前总目标函数是多项 loss 的加权和：

\[
L = \lambda_m L_{mask} + \lambda_d L_{depth} + \lambda_c L_{contour} + \lambda_p L_{prior}
\]

在代码里又拆成以下几项：

### 8.1 `mask_iou_loss`

约束：

- 渲染出来的 primitive mask
- 和输入 observed mask 的重叠程度

作用：

- 保证整体位置、尺度、覆盖区域大致对齐

局限：

- 它更偏粗轮廓重叠
- 对真实烧杯的结构细节不够敏感

---

### 8.2 `robust_depth_loss`

约束：

- rendered depth
- 和 observed depth / completed depth 的差异

作用：

- 约束三维尺度和表面深度位置

当前问题：

- 透明物体的深度本身不可靠
- `depth_completed` 也只是近似
- 所以 depth loss 是有帮助的，但不能完全依赖

---

### 8.3 `contour_chamfer_loss`

约束：

- rendered mask 的边界
- 和 observed mask 边界之间的 Chamfer 距离

作用：

- 比单纯 IoU 更关注边界对齐
- 帮助 primitive 贴近真实轮廓

---

### 8.4 `axis_angle_loss`

约束：

- rendered mask 的主轴方向
- 和 observed mask 的主轴方向

作用：

- 防止 primitive 完全转歪
- 对细长物体尤其重要

---

### 8.5 `top_width_loss`

约束：

- 渲染结果顶部宽度
- 和 observed mask 顶部宽度

作用：

- 对 beaker / flask 这类有明显开口结构的器皿有帮助

但当前只约束了一个比较粗的 top-width 标量，还不够表达完整的杯口结构。

---

### 8.6 `prior_regularization`

约束：

- 半径必须为正
- 高度必须为正
- 高宽比不能太离谱
- `roll/pitch` 不应过大

作用：

- 防止优化跑出物理上明显不合理的解

---

## 9. 当前 primitive recovery 做过的尝试

目前大体做过三轮尝试。

### 尝试 1：二维近似 primitive

最早的实现是：

- 用图像上的中心点、宽度、高度
- 直接渲染一个二维矩形 + 上下椭圆

问题：

- 本质还是 2D 近似
- 对真实 3D 位置和姿态表达不足
- 很容易出现奇怪的中轴伪影
- 拟合结果和真实烧杯差距较大

这一阶段的意义主要是：

- 跑通“初始化 → 渲染 → loss → 优化 → 输出”的代码链路

---

### 尝试 2：升级为 3D 参数化

后续把参数升级成：

\[
(x, y, z, roll, pitch, yaw, radius, height)
\]

同时把渲染器改成：

- 先在局部坐标系采样圆柱表面
- 再做 3D 旋转和平移
- 最后投影到图像

这一阶段的改进点：

- primitive 终于具备了真实 3D pose 表达能力
- 不再局限于图像平面上的二维位置调整

但结果仍然和真实烧杯差距较大。

---

### 尝试 3：加入收敛诊断

为了区分“优化结束了”和“优化真的得到了可信结果”，又加了：

- `loss_history.json`
- `convergence` 诊断字段

现在可以判断：

- 优化器是否成功结束
- loss 是否稳定
- 参数是否合理
- 视觉拟合是否可信

这一步的价值在于：

- 即使最终结果不好，也可以判断问题到底出在什么环节

---

## 10. 当前简单实验结果与现象

当前在 `primitive_outputs/` 下已经保存了若干次 beaker primitive fitting 的结果，例如：

- `beaker_fit`
- `beaker_fit_3d_000000`
- `beaker_fit_3d_000000_loss`
- `beaker_fit_3d_000000_raw_depth`

每次实验都会输出：

- `fit_summary.json`
- `loss_history.json`（新版）
- `observed_mask.png`
- `rendered_mask.png`
- `mask_overlay.png`
- `observed_depth.png`
- `rendered_depth.png`

这些结果表明：

1. primitive fitting 的代码链路已经完全可运行；
2. 现在可以拿真实的 mask + depth 做拟合；
3. 但拟合出来的形状和真实烧杯仍有明显差距。

---

## 11. 当前结果不理想的主要原因分析

### 11.1 输入 mask 质量仍然不够高

虽然已经使用 FastSAM 生成 mask，但当前 mask 仍然不是“非常干净的烧杯实例轮廓”，这会直接影响：

- 初始化
- contour loss
- mask overlap

因为 primitive 只能拟合你给它的观测，观测本身不准，后面再强的优化也会偏。

---

### 11.2 beaker 模型表达能力仍然太弱

当前 beaker 本质上仍然更接近：

- cylinder-like primitive

但真实烧杯通常有：

- 杯口外翻
- 侧壁轻微 taper
- 底部圆角
- 开口容器的结构

所以当前 primitive family 本身不够表达真实烧杯。

这意味着优化器只能在一个“错误但接近”的模型族中找最优解。

---

### 11.3 depth supervision 仍然不够强

虽然有 `depth_completed`，但透明器皿上的深度仍然不是严格真值：

- rim 会被平滑
- 内外壁可能混淆
- 深度 completion 会带入结构偏差

所以 depth consistency 当前能提供帮助，但不足以强力纠正几何。

---

### 11.4 loss 更偏向粗轮廓而不是真实容器结构

当前 loss 组合更容易让优化器去匹配：

- 大致区域覆盖
- 主轴方向
- 顶部宽度

而不是严格匹配：

- beaker 的开口结构
- 杯壁 taper
- 容器的高度轮廓

因此即使 `success = true`，最终也可能只是“数值上找到了一个局部最优解”，但这个解不一定几何上可信。

---

## 12. 当前最准确的结论

当前 primitive recovery 部分的实验最准确的定位是：

> 已经完成了一个可运行的 optimization-based primitive fitting MVP，能够把透明玻璃器皿恢复问题转化为一个“结构化参数优化问题”，并在真实 `mask + depth` 输入上输出 primitive 参数和可视化结果。但当前拟合质量仍然较粗，主要受限于输入 mask 质量、depth 可靠性以及 beaker primitive 模型表达能力不足。

也就是说：

- 这个 primitive recovery 方向是**技术上可落地的**
- 但现在还处于 **原型验证阶段**
- 还不能把当前拟合结果直接当成高质量几何恢复结果

---

## 13. 当前最值得继续推进的方向

### 13.1 先提升输入 mask 质量

这是最优先的问题。

建议：

- 手工修少量高质量 mask
- 或者对 FastSAM 结果做更强约束

因为这是所有 primitive fitting 的上游输入。

---

### 13.2 把 beaker primitive 升级成更真实的模板

建议从简单 cylinder 升级为：

- tapered open beaker
- with rim
- with rounded bottom

这样模型表达会更接近真实结构。

---

### 13.3 增强结构约束

后续应加入：

- vertical extent loss
- side wall taper constraint
- top opening shape constraint
- 更明确的 height profile / width profile consistency

---

### 13.4 扩展到更多类别

在 beaker 之后，推荐依次做：

- glass rod
- flask
- dropper bottle

这些类的 primitive 结构更能体现这个方向的价值。

---

### 13.5 与 affordance 部分融合

primitive recovery 的最终意义不只是“拟合器皿形状”，而是：

- 在 primitive surface 上定义 affordance region
- 把 2D affordance map 投影到结构化几何表面
- 生成 task-conditioned 3D contact / grasp proposal

这才是整个项目真正的闭环。

---

## 14. 最终总结

当前 primitive recovery 部分已经完成了以下关键工作：

1. 明确了 primitive recovery 的问题形式；
2. 完成了 beaker primitive 的 3D 参数化；
3. 实现了 mask/depth 驱动的 optimization fitting；
4. 实现了渲染、loss、优化、可视化和收敛诊断；
5. 在真实 `mask + depth` 输入上完成了原型级验证。

但当前也明确暴露出以下问题：

1. 输入 mask 质量决定拟合上限；
2. transparent depth 仍不是强监督；
3. 当前 beaker primitive 过于简化；
4. 数值优化结束并不等于几何上可信的拟合。

因此，当前 primitive recovery 最合适的定位是：

> 一个已经跑通的、可解释的、可继续扩展的 structured geometry recovery prototype。

它已经把“透明玻璃器皿 primitive fitting”从概念问题推进到了可执行代码层面，但距离高质量、稳定、可泛化的结构恢复系统仍有明显距离。
