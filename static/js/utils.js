/**
 * 心愈 AI - 通用工具函数
 */

/**
 * 防抖函数
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * 节流函数
 */
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * XSS 过滤
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

/**
 * 格式化日期
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

/**
 * 相对时间
 */
function timeAgo(dateString) {
    const now = new Date();
    const date = new Date(dateString);
    const diff = now - date;
    
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (days > 0) return `${days}天前`;
    if (hours > 0) return `${hours}小时前`;
    if (minutes > 0) return `${minutes}分钟前`;
    return '刚刚';
}

/**
 * 获取 URL 参数
 */
function getUrlParam(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

/**
 * 深拷贝
 */
function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

/**
 * 本地存储封装
 */
const storage = {
    set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (e) {
            console.error('Storage set error:', e);
            return false;
        }
    },
    get(key, defaultValue = null) {
        try {
            const value = localStorage.getItem(key);
            return value ? JSON.parse(value) : defaultValue;
        } catch (e) {
            console.error('Storage get error:', e);
            return defaultValue;
        }
    },
    remove(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (e) {
            console.error('Storage remove error:', e);
            return false;
        }
    }
};

/**
 * 风险等级颜色
 */
function getRiskColor(riskLevel) {
    const colors = {
        '低风险': 'background:#f4f8ec;color:#4f772d;',
        '中等风险': 'background:#faf3e0;color:#a07c3a;',
        '高风险': 'background:#fdf0ee;color:#c0392b;'
    };
    return colors[riskLevel] || 'background:#eef0e8;color:#5a7052;';
}

/**
 * 导出为 CSV
 */
function exportToCSV(data, filename) {
    if (!data || data.length === 0) return;
    
    const headers = Object.keys(data[0]);
    const csvContent = [
        headers.join(','),
        ...data.map(row => headers.map(h => row[h]).join(','))
    ].join('\n');
    
    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
}

/**
 * ── Forest UI Micro-interactions ──
 * 页面加载、卡片、按钮微交互动效
 */

// 交叉观察器：元素进入视口时触发 fade-up
function initScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });

  document.querySelectorAll('.sc, .mc, .card, .glass-card, .post-card, .record-card').forEach(el => {
    if (!el.classList.contains('anim-fade-up')) {
      el.style.opacity = '0';
      el.style.transform = 'translateY(14px)';
      el.style.transition = 'opacity 0.4s cubic-bezier(0.4,0,0.2,1), transform 0.4s cubic-bezier(0.4,0,0.2,1)';
      observer.observe(el);
    }
  });
}

// 按钮涟漪效果
function initRipple() {
  document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('click', function(e) {
      const rect = this.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const ripple = document.createElement('span');
      ripple.style.cssText = `position:absolute;width:6px;height:6px;border-radius:50%;background:rgba(255,255,255,0.35);left:${x}px;top:${y}px;transform:scale(0);animation:rippleAnim 0.5s ease-out forwards;pointer-events:none;`;
      this.style.position = 'relative';
      this.style.overflow = 'hidden';
      this.appendChild(ripple);
      setTimeout(() => ripple.remove(), 520);
    });
  });
  // 注入 ripple keyframe
  if (!document.getElementById('ripple-style')) {
    const s = document.createElement('style');
    s.id = 'ripple-style';
    s.textContent = '@keyframes rippleAnim{to{transform:scale(28);opacity:0}}';
    document.head.appendChild(s);
  }
}

// Sidebar 链接滑入指示器
function initSidebarIndicator() {
  const links = document.querySelectorAll('.sidebar-link');
  links.forEach(link => {
    link.addEventListener('mouseenter', function() {
      this.style.transition = 'all 0.12s cubic-bezier(0.4,0,0.2,1)';
    });
  });
}

// 数字计数动画
function animateCounter(el, target, duration = 800) {
  const start = 0;
  const startTime = performance.now();
  const update = (currentTime) => {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(start + (target - start) * eased);
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

// 页面加载完成后初始化所有微交互
document.addEventListener('DOMContentLoaded', () => {
  initScrollReveal();
  initRipple();
  initSidebarIndicator();

  // 数字卡片计数动画
  setTimeout(() => {
    document.querySelectorAll('.sc-val, .stat-number, .scard-val').forEach(el => {
      const val = parseInt(el.textContent);
      if (!isNaN(val) && val > 0) animateCounter(el, val, 900);
    });
  }, 300);

  // 登录成功提示
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get('login_success') === 'true') {
    setTimeout(() => {
      if (window.notificationSystem) {
        window.notificationSystem.success('登录成功，欢迎回来 🌿');
      }
    }, 400);
  }
});
