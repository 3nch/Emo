/**
 * 心愈 AI - 全局错误处理与通知系统
 * 提供统一的错误处理、Toast 提示和通知功能
 */

// 错误消息映射
const ERROR_MESSAGES = {
    400: '请求格式有误，请检查输入内容',
    401: '请先登录后再操作',
    403: '您没有权限执行此操作',
    404: '请求的资源不存在',
    408: '请求超时，请稍后重试',
    429: '请求过于频繁，请稍后再试',
    500: '服务器内部错误，请稍后重试',
    502: '服务器网关错误，请稍后重试',
    503: '服务暂时不可用，请稍后重试',
    504: '服务器响应超时，请稍后重试',
    network: '网络连接失败，请检查网络设置',
    timeout: '请求超时，请稍后重试',
    unknown: '发生未知错误，请稍后重试'
};

// 通知系统
class NotificationSystem {
    constructor() {
        this.notifications = [];
        this.container = null;
        this.checkinReminderInterval = null;
        this.communityUpdateInterval = null;
        this.init();
    }

    init() {
        this.createContainer();
        this.startCheckinReminder();
        this.startCommunityUpdates();
    }

    // 创建通知容器
    createContainer() {
        if (document.getElementById('notification-container')) return;

        const container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = 'position:fixed;top:72px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:10px;max-width:340px;width:100%;pointer-events:none;';
        document.body.appendChild(container);
        this.container = container;
    }

    // 显示通知
    show(message, type = 'info', duration = 4000) {
        const id = Date.now() + Math.random();
        const notification = { id, message, type };

        const toast = document.createElement('div');
        toast.style.cssText = this.getToastStyle(type);
        toast.id = `toast-${id}`;
        toast.innerHTML = this.getToastHTML(message, type, id);
        toast.style.transform = 'translateX(120%)';
        toast.style.opacity = '0';

        this.container.appendChild(toast);

        requestAnimationFrame(() => {
            toast.style.transition = 'transform 0.3s cubic-bezier(0.4,0,0.2,1), opacity 0.3s';
            toast.style.transform = 'translateX(0)';
            toast.style.opacity = '1';
        });

        if (duration > 0) {
            setTimeout(() => this.remove(id), duration);
        }

        this.notifications.push(notification);
        return id;
    }

    // 获取Toast内联样式
    getToastStyle(type) {
        const base = 'pointer-events:auto;padding:12px 14px;border-radius:10px;box-shadow:0 4px 16px rgba(15,23,42,0.10);border-left:4px solid;background:#fff;min-width:260px;';
        const types = {
            success: 'border-color:#10B981;',
            error:   'border-color:#EF4444;',
            warning: 'border-color:#F59E0B;',
            info:    'border-color:#31572c;'
        };
        return base + (types[type] || types.info);
    }

    // 获取Toast HTML
    getToastHTML(message, type, id) {
        const colors = { success:'#10B981', error:'#EF4444', warning:'#F59E0B', info:'#31572c' };
        const c = colors[type] || colors.info;
        const icons = {
            success: `<svg width="15" height="15" fill="none" stroke="${c}" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>`,
            error:   `<svg width="15" height="15" fill="none" stroke="${c}" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>`,
            warning: `<svg width="15" height="15" fill="none" stroke="${c}" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>`,
            info:    `<svg width="15" height="15" fill="none" stroke="${c}" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`
        };
        return `<div style="display:flex;align-items:flex-start;gap:10px;">
            <span style="flex-shrink:0;margin-top:1px;">${icons[type]||icons.info}</span>
            <p style="flex:1;font-size:13px;font-weight:500;color:#0F172A;line-height:1.5;margin:0;">${message}</p>
            <button onclick="window.notificationSystem.remove(${id})" style="background:none;border:none;cursor:pointer;color:#94A3B8;padding:0;flex-shrink:0;line-height:1;font-size:16px;">×</button>
        </div>`;
    }

    // 移除通知
    remove(id) {
        const toast = document.getElementById(`toast-${id}`);
        if (toast) {
            toast.style.transform = 'translateX(120%)';
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }
        this.notifications = this.notifications.filter(n => n.id !== id);
    }

    // 便捷方法
    success(message, duration) {
        return this.show(message, 'success', duration);
    }

    error(message, duration) {
        return this.show(message, 'error', duration);
    }

    warning(message, duration) {
        return this.show(message, 'warning', duration);
    }

    info(message, duration) {
        return this.show(message, 'info', duration);
    }

    // 打卡提醒 - 每天固定时间提醒
    startCheckinReminder() {
        // 检查是否已登录
        const userId = this.getUserId();
        if (!userId) return;

        // 设置每天晚上8点提醒打卡
        const checkAndRemind = () => {
            const now = new Date();
            const reminderHour = 20; // 晚上8点
            const reminderMinute = 0;

            if (now.getHours() === reminderHour && now.getMinutes() === reminderMinute) {
                // 检查今天是否已打卡
                this.checkTodayCheckin(userId);
            }
        };

        // 每分钟检查一次
        this.checkinReminderInterval = setInterval(checkAndRemind, 60000);
        // 页面加载时也检查一次
        checkAndRemind();
    }

    // 检查今天是否已打卡
    async checkTodayCheckin(userId) {
        try {
            const response = await fetch('/api/checkin', {
                credentials: 'same-origin'
            });
            const data = await response.json();
            if (!Array.isArray(data)) return;

            const today = new Date().toISOString().split('T')[0];
            const hasCheckedIn = data.some(checkin => {
                const checkinDate = new Date(checkin.created_at).toISOString().split('T')[0];
                return checkinDate === today;
            });

            if (!hasCheckedIn) {
                this.show('🌟 晚安！记得今天的情绪打卡哦～', 'info', 10000);
            }
        } catch (e) {
            console.log('Checkin reminder error:', e);
        }
    }

    // 社区更新通知
    startCommunityUpdates() {
        const userId = this.getUserId();
        if (!userId) return;

        // 每5分钟检查一次社区更新
        this.communityUpdateInterval = setInterval(() => {
            this.checkCommunityUpdates(userId);
        }, 300000); // 5分钟

        // 首次加载时检查
        this.checkCommunityUpdates(userId);
    }

    // 检查社区更新
    async checkCommunityUpdates(userId) {
        try {
            // 获取当前时间5分钟前的时间戳
            const fiveMinutesAgo = new Date(Date.now() - 5 * 60000).toISOString();

            const response = await fetch('/api/posts', {
                credentials: 'same-origin'
            });

            if (!response.ok) return;

            const posts = await response.json();
            if (!Array.isArray(posts)) return;
            
            const newPosts = posts.filter(post => {
                return new Date(post.created_at) > new Date(fiveMinutesAgo);
            });

            if (newPosts.length > 0) {
                // 显示社区更新通知
                this.show(`📢 社区有 ${newPosts.length} 条新帖子，点击查看`, 'info', 8000);
            }
        } catch (e) {
            console.log('Community update check error:', e);
        }
    }

    // 获取用户ID
    getUserId() {
        // 从sessionStorage获取用户ID（前端存储）
        return sessionStorage.getItem('user_id');
    }

    // 清理定时器
    destroy() {
        if (this.checkinReminderInterval) {
            clearInterval(this.checkinReminderInterval);
        }
        if (this.communityUpdateInterval) {
            clearInterval(this.communityUpdateInterval);
        }
    }
}

// API 错误处理工具
const APIErrorHandler = {
    /**
     * 处理API响应错误
     */
    handle: async function(response, customMessage) {
        let errorMessage = customMessage;

        if (!errorMessage) {
            // 尝试从响应中获取错误消息
            try {
                const data = await response.clone().json();
                errorMessage = data.error || data.message || ERROR_MESSAGES[response.status] || ERROR_MESSAGES.unknown;
            } catch (e) {
                errorMessage = ERROR_MESSAGES[response.status] || ERROR_MESSAGES.unknown;
            }
        }

        // 显示错误提示
        notificationSystem.error(errorMessage);

        // 如果是401未授权，可能需要跳转登录
        if (response.status === 401) {
            setTimeout(() => {
                window.location.href = '/login?redirect=' + encodeURIComponent(window.location.pathname);
            }, 1500);
        }

        // 如果是403禁止访问
        if (response.status === 403) {
            setTimeout(() => {
                window.location.href = '/';
            }, 1500);
        }

        return { error: true, message: errorMessage, status: response.status };
    },

    /**
     * 包装fetch请求，自动处理错误
     */
    fetch: async function(url, options = {}) {
        try {
            const response = await fetch(url, {
                ...options,
                credentials: 'same-origin'
            });

            // 检查是否是错误状态码
            if (!response.ok) {
                return await this.handle(response);
            }

            return { error: false, response };
        } catch (error) {
            // 网络错误
            const errorMsg = ERROR_MESSAGES.network;
            notificationSystem.error(errorMsg);
            return { error: true, message: errorMsg, network: true };
        }
    }
};

// 全局实例
let notificationSystem;

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    notificationSystem = new NotificationSystem();

    // 将实例挂载到window对象，便于全局访问
    window.notificationSystem = notificationSystem;
    window.APIErrorHandler = APIErrorHandler;

    // 设置全局fetch错误处理
    const originalFetch = window.fetch;
    window.fetch = async function(...args) {
        try {
            const response = await originalFetch.apply(this, args);

            // 如果是API请求且返回错误，自动处理
            const url = args[0];
            if (typeof url === 'string' && url.startsWith('/api/')) {
                if (!response.ok) {
                    // 不自动显示错误，留给调用方处理
                    return response;
                }
            }

            return response;
        } catch (error) {
            // 网络错误
            if (notificationSystem) {
                notificationSystem.error(ERROR_MESSAGES.network);
            }
            throw error;
        }
    };

    // 页面卸载时清理
    window.addEventListener('beforeunload', function() {
        if (notificationSystem) {
            notificationSystem.destroy();
        }
    });
});

// 导出以便其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NotificationSystem, APIErrorHandler, ERROR_MESSAGES };
}
