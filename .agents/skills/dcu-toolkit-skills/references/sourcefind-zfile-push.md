# Sourcefind 分发站 / Git 仓库操作要点（DCU 工具链）

> 运维级知识库：dcu-toolkit 包从 sourcefind 拉取 DTK / 推送 skill 包时的可靠手法。
> 不属于"文档边界纪律"（那部分在 ~/.hermes/notes/dcu-toolkit-doc-discipline.md），
> 这里只记**可复跑的技术手法**。

## 1. download.sourcefind.cn 是 z-file，curl 直链规律

- 站点 `https://download.sourcefind.cn:65024` 是 **z-file 文件管理器**，页面靠 JS 渲染（SPA）。
- **目录/网页路径（拉不到正文）**：
  - `https://download.sourcefind.cn:65024/1/main/latest` → 返回 SPA 外壳（~432 字节 HTML，无文件内容）
  - `https://download.sourcefind.cn:65024/1/main/DTK与驱动版本配套关系表.md` → 同样返回 SPA 壳，curl 拿不到表格
- **真实文件直链（curl 可读）**：
  - 格式：`https://download.sourcefind.cn:65024/file/1/<文件名>`
  - 例：`https://download.sourcefind.cn:65024/file/1/DTK与驱动版本配套关系表.md`
  - 实测：`curl -sL` 返回 3839 字节 UTF-8 表格文本（HTTP 200），无需浏览器渲染。
  - 中文文件名可原样（不转义）或 `%E4%B8%8E...` 转义，两者均 200。
- **结论**：任何"从 sourcefind 下载站拉取文件内容"的需求，必须用 `/file/1/<名>` 直链，
  不要用 `/1/main/...` 网页路径（那是给人浏览器看的，curl 读不到）。

## 2. DTK 配套关系表"联网复核"的可执行写法

- 配套表直链可 curl，所以"联网复核"对 agent 是**真实可行**的（不是空头承诺）：
  ```bash
  curl -sL "https://download.sourcefind.cn:65024/file/1/DTK与驱动版本配套关系表.md"
  # 解析返回的 Markdown 表格，与本 skill 第5节快照比对；远程出现新版本/要求变更以远程为准
  ```
- 降级：若运行环境无外网（纯内网隔离），则以 skill 内快照为基线，并在输出显式提示
  "配套表未联网复核，版本以快照为准，请人工确认"。
- **绝不臆造版本号**：若某文件直链也拉不到（如 latest 目录下列表 API 不通），
  宁可放"结构框架 + 链接 + 联网复核说明"，也不要编造型号/版本数据。

## 3. sourcefind Git 推送（skill 包提交）

- 本地无 `~/.git-credentials` 的 sourcefind 条目，也没配 credential.helper → HTTPS push 需用户名/密码。
- **可靠做法：token 拼进远程 URL 一次性 push**（不在命令行外落盘、不写 memory、不进包）：
  ```bash
  TOKEN="<sourcefind_access_token>"
  URL="https://jixx:${TOKEN}@developer.sourcefind.cn/codes/jixx/dcu-toolkit-skills.git"
  git push "$URL" master
  ```
- **偶发 Access denied**：读权（ls-remote / clone）正常、写权（push）被拒 → 换新 access token 重试，
  不用排查命令/编码。token 在 ~/.hermes/notes/ 或用户处，不进包。
- 服务器出口 IP 不稳定（多 IP 需白名单），若 push 卡在网络层，确认白名单而非改命令。

## 4. 提交纪律（用户明确要求）

- 每个经用户审查的改动**直接提交并 push**（用户原话："修改好了之后直接提交" / "push 了吗"）。
- 只提交本次实际改动的文件（`git add <具体文件>`，不盲目 `git add -A` 带入未审查改动）；
  但本包常需多文件联动提交时，确认范围后提交。
- install.sh 的子 skill 校验数组（SUBS）新增子 skill 时必须同步加，否则 clone 后校验报错。

## 5. 验证证据

- 改完脚本/文档后，用临时脚本 `/tmp/hermes-verify-*.sh` 实测（含真实 clone / 真实 curl 拉表），
  跑完即删；明确标注为 **ad-hoc 验证**，非套件绿。
- 例：install.sh 改 SUBS 后，用临时 HOME 真实 clone 验证「子 skill 数量齐全（当前 10 个）」再 push。
