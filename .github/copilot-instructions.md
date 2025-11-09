## 快速指南（针对 AI 协作代理）

下面的说明帮助你在本仓库中快速开展编码任务：构建/运行/调试，以及遵循本项目的显性约定。

- 项目结构要点（高层）
  - 代码按主题组织在顶级目录：`container/`, `graph/`, `search/`, `sort/`, `SUDA_TEST/`。
  - 每个模块通常包含实现文件（例如 `AVL.cpp`, `BST.cpp`, `KMP.cpp`）和一个 `test/` 子目录，测试文件是独立的小程序（单文件 main），例如 `container/test/linked_list_test.cpp`。
  - 预编译产物放在 `build/linux/x86_64/release/`（比如 `arr`, `node`, `stack`, `tree` 等二进制）。这些文件可能是历史输出，优先使用源码构建工具再运行。

- 构建与运行（可执行流程）
  - 本仓库包含 `xmake.lua`：推荐使用 xmake 来构建项目。`xmake.lua` 定义了若干 target：`arr`, `node`, `stack`, `tree`, `graph`, `sort`（见 `xmake.lua` 顶部）。
    - 常见操作（xmake 的标准用法，可在 agent 建议中直接引用）：
      - 构建所有 targets：`xmake`（在仓库根目录运行）
      - 运行单个 target（xmake 通常支持）：`xmake run <target>`（例如 `xmake run arr`）
  - 另外：VS Code 提供了一个 C/C++ 任务（`C/C++: g++ 生成活动文件`）用于编译当前活动文件为同名可执行文件；它仅编译单个源文件（适合快速验证小改动）。
  - 若使用 VS Code 调试，`.vscode/launch.json` 的默认 `program` 被设为 `${workspaceFolder}/a.out`。要调试由 xmake 或其他方式生成的具体二进制，更新 `program` 为 `build/linux/x86_64/release/<target>` 或你当前生成的可执行文件路径。

- 测试约定与样例
  - “测试”通常以小的可执行程序存在（非 GoogleTest 等框架）。测试文件位置举例：
    - `container/test/seq_list_test.cpp`
    - `graph/test/Graph_test.cpp`
    - `search/test/search_test.cpp`
    - `sort/test/sort_test.cpp`
  - 如果新增一个算法或数据结构，遵循现有模式：在对应模块下添加实现文件（例如 `search/MyAlgo.cpp`）并在对应 `test/` 下添加一个独立的测试 runner，然后将其加入 `xmake.lua`（通过 `add_files()`）以便由 xmake 管理。

- 代码风格与约定（可被发现的规则）
  - 文件以单个类/算法实现为主（多数为单文件 .cpp 实现）。
  - 设计上偏向教学/练手格式：实现 + 简单测试 runner 而非可复用库接口。
  - 改动慎重：请尽量不破坏现有测试 runner 的命令行签名（通常无参数或少量参数）。

- 集成点与注意事项
  - 目录 `build/` 包含历史输出，不能保证与源码同步——不要把它当作权威构建来源；优先使用 `xmake.lua` 或按需使用 VS Code 的单文件编译任务。
  - 若需要引入第三方依赖或测试框架：先在顶层说明（README 或额外构建脚本）并将 `xmake.lua` 一并更新，保持仓库当前的构建方式一致。

- 示例片段（用于 agent 建议）
  - 要构建并运行 `graph` target，建议的步骤：
    1. 在仓库根目录运行 `xmake`（构建）
    2. 运行 `xmake run graph` 或直接运行生成的二进制（`build/linux/x86_64/release/graph`）
  - 要调试某个生成的可执行：将 `.vscode/launch.json` 中的 `program` 修改为对应二进制路径并使用现有配置启动调试。

如果某部分信息不完整或你希望把额外的本地习惯（例如常用编译选项、shell alias、或 CI 流程）写入这个文件，请告诉我要加入的细节，我会根据你的反馈更新此文档。

-- 自动生成（由仓库扫描得出）
