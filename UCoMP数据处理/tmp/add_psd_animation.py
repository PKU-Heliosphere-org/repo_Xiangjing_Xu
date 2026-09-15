import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / '工作总结/内外传功率比分布.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))
heading = '### 内外传分块功率谱动画（5×5 像素）'
if not any(heading in ''.join(cell.get('source', [])) for cell in nb['cells']):
    markdown = heading + '''

使用已计算的 `psd_in/out(x,y,t,k,omega)`，无需重新运行小波变换。
选择原始像素 x[10:40]、y[20:100]，按 5×5 平均得到 6×16 个子图。
内传显示在负 k，外传显示在正 k，两个 imshow 之间保留未计算的空白区。
所有帧及两支共享色标；标题时间相对于输入第一帧。
本单元仅构建动画，不自动写出大量图片或视频。
'''
    code = '''from pathlib import Path
import importlib.util

# 支持以工作总结目录或项目根目录作为 notebook 工作目录。
animation_module_path = Path('animate_directional_psd.py')
if not animation_module_path.is_file():
    animation_module_path = Path('工作总结/animate_directional_psd.py')
spec = importlib.util.spec_from_file_location('directional_psd_animation', animation_module_path)
animation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(animation_module)

# 重运行单元时关闭旧窗口，并保留动画对象以避免被回收。
if 'fig_psd_blocks' in globals():
    plt.close(fig_psd_blocks)
fig_psd_blocks, ani_psd_blocks = animation_module.animate_directional_psd(
    psd_in, psd_out, k_array, omega_array, dt=30.0, interval=150,
)
plt.show()

# 可选：导出 GIF（启用后可能需要较长时间）。
# ani_psd_blocks.save('psd_blocks_in_out.gif', writer='pillow', fps=7, dpi=80)
'''
    nb['cells'].extend([
        dict(cell_type='markdown', metadata={}, source=markdown.splitlines(keepends=True)),
        dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
             source=code.splitlines(keepends=True)),
    ])
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

