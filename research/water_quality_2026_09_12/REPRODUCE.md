# 复核方法

从 `D:\WDN\quantum` 的 PowerShell 运行。以下命令只向本研究目录写入构建/实验结果，不修改原始输入或源码。

```powershell
python research/water_quality_2026_09_12/audit_inputs.py
cmake -S EPANET-dev -B research/water_quality_2026_09_12/build_epanet -DBUILD_TESTS=OFF
cmake --build research/water_quality_2026_09_12/build_epanet --config RelWithDebInfo --target epanet2
python research/water_quality_2026_09_12/pilot_quality.py
python research/water_quality_2026_09_12/cvpipe_probe/cvpipe_probe.py
```

环境：本机Anaconda Python，`numpy`与`PyMuPDF/fitz`；MSVC 19.50.35724.0，Windows SDK 10.0.26100.0，CMake默认Visual Studio 18 2026生成器。`RelWithDebInfo`的C编译配置含`/O2 /Ob1`。原CMake在MSVC下将Release flags设置为`/GL`，为避免把缺少明确速度优化的构建当性能基线，本实验最终统一使用RelWithDebInfo。未运行完整Boost/nrtest测试套件；没有对现有求解器做代码修复。

主清单为`input_inventory.json`；敏感性实验为`pilot_quality_results.json`和`pilot_runs/*.npz`；CVPIPE独立复现为`cvpipe_probe/cvpipe_probe_results.json`、逐水质步CSV与原始生成INP/RPT。二者均使用同一DLL指纹，具体指纹记在各自JSON内。

敏感性实验协议：每网水力求解一次；8组步长/容差；一次含节点提取的轨迹记录，再做3次计时。计时仅覆盖`runQ/nextQ`循环（含Python调用与引擎收支计算），不含打开/初始化、输入解析、水力与结果提取。各组观察时间网格严格检查一致。Net1额外新建独立Project复查300s/0.01与2s/1e-4，分别与原记录全部时间及浓度完全一致，排除了多次初始化状态污染这一疑点。

较细参考为2s/1e-4；它不是解析真值，也没有完成连续方程总误差认证。目录中早期极端容差试跑留下的1e-5 NPZ不属于最终JSON规定的8组结果；它们不用于最终表格与计时。关于质量比、最大差异与公平成本比较的解释见主报告。

CVPIPE13个案例为独立最小验证，与四网敏感性实验分开；使用单管解析旅行时间和一级反应解，改变长度/步长/有无反应，并保留重复轨迹检查。结论只针对本地版本20306的当前源码与DLL，不断言其他发行版本是否有此问题。
