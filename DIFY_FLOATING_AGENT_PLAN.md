# Dify 悬浮智能体小助手实现计划（落地版）

## 1. 目标与范围

- 在现有 Flask 项目中接入一个**全局悬浮智能体小助手**。
- 智能体通过后端代理调用你已配置好的 Dify Agent API。
- 交互要求：
  - 可爱的卡通形象（悬浮在页面右下角）
  - 点击展开聊天面板
  - 支持发送消息、接收回复、快捷提问
  - 最小化/关闭
- 本期先做稳定 MVP，后续再加流式输出与语音输入。

---

## 2. 架构设计

### 2.1 调用链路

1. 前端悬浮助手组件发起请求到项目后端：`/api/assistant/chat`
2. 后端读取环境变量中的 Dify Key，并转发到 Dify API
3. Dify 返回结果后，后端清洗并返回前端
4. 前端渲染消息气泡，并缓存会话 ID 与最近对话

### 2.2 安全原则

- **前端不直接调用 Dify**，避免泄露密钥。
- Dify Key 仅保存在服务端环境变量。
- 对外只暴露项目自己的后端接口。

---

## 3. 文件级改造清单（按你当前项目结构）

## 3.1 后端

### A. `app.py`（新增 API）

新增接口：`POST /api/assistant/chat`

功能：
- 接收参数：`message`、`conversation_id`（可空）
- 校验空消息
- 读取配置：`DIFY_API_URL`、`DIFY_API_KEY`
- 调用 Dify（`/v1/chat-messages`）
- 解析并返回：
  - `reply`
  - `conversation_id`
  - `success`
  - `error`（失败时）

可选：
- 记录日志（请求耗时、状态码）
- 风险词命中后附加本地提示

### B. `config.py`（新增配置项）

新增：
- `DIFY_API_URL`
- `DIFY_API_KEY`
- `DIFY_TIMEOUT`（如 30 秒）

来源：环境变量优先。

---

## 3.2 前端

### C. `templates/base.html`（全局挂载）

新增：
- 悬浮助手容器（固定定位右下）
- 聊天面板（标题区、消息区、输入区）
- 卡通头像（SVG 或 PNG）

建议结构：
- `#assistant-fab`：悬浮按钮
- `#assistant-panel`：聊天弹层
- `#assistant-messages`：消息列表
- `#assistant-input`：输入框

### D. `static/js/assistant.js`（新建）

实现：
- 展开/收起
- 消息渲染（用户/助手）
- 请求 `/api/assistant/chat`
- 维护 `conversation_id`
- 本地缓存最近 N 条消息
- 错误提示（超时/失败）

### E. `static/css/assistant.css`（新建）

实现：
- 可爱卡通风格
- 呼吸动画、hover 动画
- 聊天气泡样式
- 面板过渡动画
- 移动端适配（宽度、高度、输入区）

---

## 4. 里程碑计划

| 里程碑 | 内容 | 产出 | 工期 |
|---|---|---|---|
| M1 | 后端 Dify 代理 API | `/api/assistant/chat` 可通 | 0.5 天 |
| M2 | 悬浮按钮 + 面板骨架 | 全局可打开助手 | 0.5 天 |
| M3 | 聊天收发 + 会话 ID | 可连续对话 | 0.5 天 |
| M4 | 卡通形象 + 动效 | 视觉完成度提升 | 0.5 天 |
| M5 | 异常处理 + 测试 | 稳定上线版本 | 0.5 天 |

总计：约 2~3 天。

---

## 5. 详细任务拆解（WBS）

- [ ] 新增配置项并读取环境变量（`config.py`）
- [ ] 在 `app.py` 新增 `/api/assistant/chat`
- [ ] Dify 请求封装与返回清洗
- [ ] 在 `base.html` 注入全局助手容器
- [ ] 新建 `assistant.css` 完成样式
- [ ] 新建 `assistant.js` 完成交互
- [ ] 接入快捷问题（3~5 条）
- [ ] 对话历史本地缓存（`localStorage`）
- [ ] 异常提示（网络失败、超时）
- [ ] 联调与验收测试

---

## 6. API 约定（项目内）

### 请求

`POST /api/assistant/chat`

```json
{
  "message": "我今天有点焦虑",
  "conversation_id": "optional-id"
}
```

### 响应（成功）

```json
{
  "success": true,
  "reply": "我在这里陪你，我们可以先从呼吸开始。",
  "conversation_id": "new-or-old-id"
}
```

### 响应（失败）

```json
{
  "success": false,
  "error": "Dify request failed"
}
```

---

## 7. 验收标准（DoD）

- [ ] 页面右下角显示可爱卡通悬浮助手
- [ ] 点击可展开/收起聊天面板
- [ ] 前端可调用后端接口并获得 Dify 回复
- [ ] 支持连续会话（`conversation_id` 持续）
- [ ] 页面刷新后可恢复最近会话（可选 N 条）
- [ ] 接口失败时有友好提示，不影响主流程
- [ ] 移动端可正常使用

---

## 8. 后续增强（第二期）

- 流式输出（SSE）
- 语音输入 / TTS 播报
- 拖拽悬浮球与吸边
- 与检测结果联动（如报告页快捷问答）
- 多角色皮肤（治愈猫咪/柴犬/树洞精灵）
