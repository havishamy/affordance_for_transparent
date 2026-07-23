# `affordance/infer_chem_hova.py` 对接说明

这个脚本用于**单张 RGB 图像 + 自然语言指令**的化学场景 affordance 推理。

## 输入

命令行参数：

- `--checkpoint`：训练好的模型权重，通常是 `best.pt`
- `--rgb`：输入 RGB 图片路径，脚本用 OpenCV 读取
- `--instruction`：自然语言任务描述，例如“应该抓哪里/应该接触哪里”
- `--output-dir`：结果保存目录，默认 `affordance_outputs/chem_hova`
- `--device`：`cuda` 或 `cpu`，默认优先用 GPU

### 输入要求

- 只支持**单张图**推理，不是批处理脚本
- 图片必须能被 OpenCV 正常读取
- checkpoint 里需要包含 `model_state_dict`，最好还带上 `model_kwargs`、`text_model_name`、`image_size`

## 输出

脚本会在 `output-dir` 下生成 3 个文件：

- `affordance_heatmap.npy`：原图尺寸的 heatmap，`float32`
- `affordance_heatmap.png`：heatmap 可视化图，灰度图
- `affordance_overlay.png`：原图 + heatmap 叠加图，并在最大响应点画白点

同时终端会打印：

- 输出目录
- 预测的 top affordance point `(x, y)`

## 推理逻辑

1. 读取 RGB 图像
2. 按 checkpoint 中的 `image_size` resize
3. 用文本编码器编码 `instruction`
4. 模型输出 heatmap logits
5. sigmoid 后 resize 回原图大小
6. 取 heatmap 最大值位置作为最终点

## 注意事项

- 脚本只做**全图级别**推理，没有 mask / depth 输入
- 输出点是 heatmap 的 `argmax`，结果对指令文本很敏感
- 图像会先缩放再推理，所以最终点是映射回原图后的坐标
- 依赖项目约定的 Python 环境：`fastsam-annot`

## 示例

```bash
conda activate fastsam-annot
python -m affordance.infer_chem_hova \
  --checkpoint /home/dsj/FastSAM/affordance_runs/chem_hova_fullimg/best.pt \
  --rgb /home/dsj/FastSAM/chem_hova_dataset/images/test/ce_000006.jpg \
  --instruction "Where should I interact with the Alcohol lamp to hold it?" \
  --output-dir /home/dsj/FastSAM/affordance_outputs/chem_hova_test \
  --device cuda
```
