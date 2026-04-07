(function(){
  const widget = document.getElementById('assistant-widget');
  const fab = document.getElementById('assistant-fab');
  const panel = document.getElementById('assistant-panel');
  const closeBtn = document.getElementById('assistant-close');
  const newChatBtn = document.getElementById('assistant-new-chat');
  const stopBtn = document.getElementById('assistant-stop');
  const statusText = document.getElementById('assistant-status-text');
  const charCount = document.getElementById('assistant-char-count');
  const input = document.getElementById('assistant-input');
  const sendBtn = document.getElementById('assistant-send');
  const messages = document.getElementById('assistant-messages');

  if (!widget || !fab || !panel || !input || !sendBtn || !messages) return;

  let conversationId = localStorage.getItem('assistant_conversation_id') || '';
  let currentAbortController = null;

  const STORAGE_KEY = 'assistant_recent_messages';
  const POS_KEY = 'assistant_widget_position';
  const INITIAL_MESSAGES_HTML = messages.innerHTML;
  const MAX_CHARS = Number(input.getAttribute('maxlength') || 500);

  function setStatus(text){
    if (statusText) statusText.textContent = text;
  }

  function updateCharCount(){
    if (!charCount) return;
    const current = input.value.length;
    charCount.textContent = `${current} / ${MAX_CHARS}`;
  }

  function setSendingState(sending){
    sendBtn.disabled = sending;
    input.disabled = sending;
    if (stopBtn) stopBtn.style.display = sending ? 'inline-flex' : 'none';
    setStatus(sending ? '正在认真听你说…' : '随时陪你聊聊');
  }

  function setPanel(open){
    panel.style.display = open ? 'flex' : 'none';
    if (open) input.focus();
  }

  function scrollToBottom(){
    messages.scrollTop = messages.scrollHeight;
  }

  function appendMessage(text, role){
    const item = document.createElement('div');
    item.className = `assistant-msg ${role}`;
    item.textContent = text;
    messages.appendChild(item);
    scrollToBottom();
    return item;
  }

  function appendInfoMessage(text){
    const item = document.createElement('div');
    item.className = 'assistant-inline-tip';
    item.textContent = text;
    messages.appendChild(item);
    scrollToBottom();
    return item;
  }

  function saveRecent(){
    const items = [...messages.querySelectorAll('.assistant-msg')].slice(-14).map(el => ({
      role: el.classList.contains('user') ? 'user' : 'bot',
      text: el.textContent
    }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  }

  function restoreRecent(){
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try{
      const items = JSON.parse(raw);
      if (!Array.isArray(items) || !items.length) return;
      messages.innerHTML = '';
      items.forEach(i => appendMessage(i.text, i.role));
    }catch(_e){}
  }

  function resetConversationUI(showTip = true){
    conversationId = '';
    localStorage.removeItem('assistant_conversation_id');
    localStorage.removeItem(STORAGE_KEY);
    messages.innerHTML = INITIAL_MESSAGES_HTML;
    if (showTip) appendInfoMessage('已为你开启一个新的话题，你可以重新开始聊。');
    setStatus('新的聊天已准备好');
    scrollToBottom();
  }

  async function sendMessage(message){
    if (!message || currentAbortController) return;
    const cleanMessage = message.slice(0, MAX_CHARS).trim();
    if (!cleanMessage) return;

    appendMessage(cleanMessage, 'user');
    input.value = '';
    updateCharCount();

    const botNode = appendMessage('', 'bot');
    botNode.textContent = '思考中';
    let dotCount = 0;
    const dotTimer = setInterval(() => {
      dotCount = (dotCount + 1) % 4;
      botNode.textContent = `思考中${'.'.repeat(dotCount)}`;
    }, 300);

    currentAbortController = new AbortController();
    setSendingState(true);

    try{
      const res = await fetch('/api/assistant/chat/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          message: cleanMessage,
          conversation_id: conversationId || null
        }),
        signal: currentAbortController.signal
      });

      if (!res.ok || !res.body) {
        clearInterval(dotTimer);
        let errMsg = '暂时无法连接智能体，请稍后再试。';
        try {
          const data = await res.json();
          errMsg = data.error || errMsg;
        } catch (_e) {}

        if (String(errMsg).toLowerCase().includes('conversation not exists')) {
          conversationId = '';
          localStorage.removeItem('assistant_conversation_id');
        }

        botNode.textContent = errMsg;
        saveRecent();
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let hasToken = false;
      let hasError = false;

      const flushChunk = (chunk) => {
        buffer += chunk;
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() || '';

        blocks.forEach(block => {
          const lines = block.split('\n');
          let evt = 'message';
          let payload = '';
          lines.forEach(line => {
            if (line.startsWith('event:')) evt = line.slice(6).trim();
            if (line.startsWith('data:')) payload += line.slice(5).trim();
          });

          if (!payload) return;

          try{
            const data = JSON.parse(payload);
            if (evt === 'token') {
              if (!hasToken) {
                clearInterval(dotTimer);
                botNode.textContent = '';
                hasToken = true;
              }
              botNode.textContent += data.text || '';
              scrollToBottom();
            } else if (evt === 'done') {
              if (data.conversation_id) {
                conversationId = data.conversation_id;
                localStorage.setItem('assistant_conversation_id', conversationId);
              }
            } else if (evt === 'error') {
              clearInterval(dotTimer);
              hasError = true;
              const err = data.error || '系统繁忙，请稍后重试。';
              if (String(err).toLowerCase().includes('conversation not exists')) {
                conversationId = '';
                localStorage.removeItem('assistant_conversation_id');
              }
              botNode.textContent = err;
            }
          }catch(_e){}
        });
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        flushChunk(decoder.decode(value, { stream: true }));
      }

      clearInterval(dotTimer);
      if (!hasToken && !hasError && !botNode.textContent.trim()) {
        botNode.textContent = '我收到了你的消息，但暂时没有生成内容。';
      }
      saveRecent();
    } catch (err) {
      clearInterval(dotTimer);
      if (err && err.name === 'AbortError') {
        botNode.textContent = `${botNode.textContent || ''}\n\n（已停止生成）`.trim();
      } else {
        botNode.textContent = '网络不稳定，请稍后重试。';
      }
      saveRecent();
    } finally {
      currentAbortController = null;
      setSendingState(false);
    }
  }

  function applySnap(){
    const rect = widget.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const nextLeft = rect.left + rect.width / 2 < vw / 2 ? 12 : vw - rect.width - 12;
    const nextTop = Math.min(vh - rect.height - 12, Math.max(12, rect.top));
    widget.style.left = `${nextLeft}px`;
    widget.style.top = `${nextTop}px`;
    widget.style.right = 'auto';
    widget.style.bottom = 'auto';
    localStorage.setItem(POS_KEY, JSON.stringify({left: nextLeft, top: nextTop}));
  }

  (function restorePosition(){
    const raw = localStorage.getItem(POS_KEY);
    if (!raw) return;
    try{
      const pos = JSON.parse(raw);
      if (typeof pos.left === 'number' && typeof pos.top === 'number') {
        widget.style.left = `${pos.left}px`;
        widget.style.top = `${pos.top}px`;
        widget.style.right = 'auto';
        widget.style.bottom = 'auto';
      }
    }catch(_e){}
  })();

  let dragging = false;
  let moved = false;
  let startX = 0;
  let startY = 0;
  let baseLeft = 0;
  let baseTop = 0;

  fab.addEventListener('pointerdown', (e) => {
    dragging = true;
    moved = false;
    fab.setPointerCapture(e.pointerId);
    const rect = widget.getBoundingClientRect();
    baseLeft = rect.left;
    baseTop = rect.top;
    startX = e.clientX;
    startY = e.clientY;
  });

  fab.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true;
    widget.style.left = `${baseLeft + dx}px`;
    widget.style.top = `${baseTop + dy}px`;
    widget.style.right = 'auto';
    widget.style.bottom = 'auto';
  });

  fab.addEventListener('pointerup', (e) => {
    if (!dragging) return;
    dragging = false;
    fab.releasePointerCapture(e.pointerId);
    if (moved) {
      applySnap();
      return;
    }
    const open = panel.style.display !== 'none';
    setPanel(!open);
  });

  if (closeBtn) closeBtn.addEventListener('click', () => setPanel(false));
  if (newChatBtn) newChatBtn.addEventListener('click', () => resetConversationUI(true));
  if (stopBtn) stopBtn.addEventListener('click', () => {
    if (currentAbortController) currentAbortController.abort();
  });

  sendBtn.addEventListener('click', () => sendMessage(input.value.trim()));
  input.addEventListener('input', updateCharCount);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage(input.value.trim());
  });

  document.querySelectorAll('.assistant-quick, .assistant-mood').forEach(btn => {
    btn.addEventListener('click', () => {
      const q = btn.getAttribute('data-q') || '';
      sendMessage(q);
    });
  });

  messages.addEventListener('dblclick', (e) => {
    const target = e.target.closest('.assistant-msg.bot');
    if (!target || !navigator.clipboard) return;
    navigator.clipboard.writeText(target.textContent || '').then(() => {
      setStatus('已复制这段回复');
      setTimeout(() => setStatus('随时陪你聊聊'), 1200);
    }).catch(() => {});
  });

  window.addEventListener('resize', () => {
    const rect = widget.getBoundingClientRect();
    if (rect.right > window.innerWidth || rect.bottom > window.innerHeight) {
      applySnap();
    }
  });

  restoreRecent();
  updateCharCount();
})();
