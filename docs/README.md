# docs/ — Hugo 站点根目录

这个目录是 Hugo 的**项目根**（`hugo.yaml` 在这里），不是构建产物目录。

## 什么入库、什么不入库

**入库（脚手架，手工维护）**

```
docs/hugo.yaml                            # 站点配置
docs/layouts/list.html                    # fork 自 PaperMod，打了 5 处 PATCH（见文件头注释）
docs/layouts/_partials/entry_card.html    # 一张卡片（标题/摘要/日期/角标），首页与 Above All 共用
docs/layouts/_partials/outline_blocks.html  # Above All 页按一级专题分组的块
docs/layouts/_markup/render-passthrough.html  # 公式：构建期调 transform.ToMath 渲染
docs/layouts/_partials/extend_head.html   # KaTeX 样式表（公式的 HTML 部分靠它排版）
docs/assets/css/extended/pin-cards.css    # 置顶/一级/二级卡片角标配色 + 缺图、公式失败的占位样式
docs/assets/css/extended/outline-blocks.css  # Above All 分组块的排版
docs/themes/PaperMod                      # 主题，git submodule
```

**不入库（生成物，已在 .gitignore 里）**

```
docs/content/     # 由 tools/sync.py 从笔记生成
docs/public/      # hugo 构建输出
docs/resources/
```

`docs/content/` 完全由 `tools/sync.py` 生成，每个文件头部都有 `DO NOT EDIT` 标记；转换器在下一轮会删掉不再发布、但带这个标记的文件。**不要手改，改了会被覆盖。**

## 本地预览

```bash
git submodule update --init --recursive         # 首次：拉 PaperMod
python3 tools/sync.py && hugo serve --source docs
```

站点挂在 `/llm/` 子路径下，所以本地地址是 <http://localhost:1313/llm/>，不是根路径 —— 这一点和生产环境一致，图片和内部链接的路径问题都能在本地复现出来。

## 部署

push 到 `main` 触发 `.github/workflows/gh-pages.yaml`：装 Python → `tools/sync.py` 生成内容 → 装 Hugo → `hugo --source docs` 构建 → 发布到 GitHub Pages。

> ⚠️ **仓库设置里 Pages 的 Source 必须选 "GitHub Actions"，不要选 "Deploy from a branch → /docs"。**
> 后者会把 Hugo 的**源码树**（未渲染的 markdown、`hugo.yaml`、模板）当作静态文件直接发布出去。
