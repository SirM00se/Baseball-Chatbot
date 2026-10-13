const input = document.getElementById("message");
const chatbox = document.getElementById("chatbox");
const button = document.getElementById("button");

// Persist or initialize a unique session ID in sessionStorage
let sessionId = sessionStorage.getItem("chat_session_id");
if (!sessionId) {
    sessionId = "session_" + Math.random().toString(36).substring(2, 11) + "_" + Date.now();
    sessionStorage.setItem("chat_session_id", sessionId);
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function formatMessage(text) {
    if (!text) return "";

    // Escape HTML first to prevent XSS
    let escaped = escapeHtml(text);

    // 1. Convert Markdown links: [text](https://...)
    escaped = escaped.replace(/\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g, (match, linkText, url) => {
        return `<a href="${url}" target="_blank" rel="noopener noreferrer">${linkText}</a>`;
    });

    // 2. Convert standalone URLs that are not already part of an <a> tag
    escaped = escaped.replace(/(<a\b[^>]*>.*?<\/a>)|(https?:\/\/[^\s<"'\u3010\u3011]+)/g, (match, existingLink, url) => {
        if (existingLink) {
            return existingLink;
        }

        let cleanUrl = url;
        let trailing = "";

        while (/[.,;:!?)\]\u3010\u3011]$/.test(cleanUrl) || cleanUrl.endsWith("&gt;")) {
            if (cleanUrl.endsWith("&gt;")) {
                trailing = "&gt;" + trailing;
                cleanUrl = cleanUrl.slice(0, -4);
            } else {
                trailing = cleanUrl.slice(-1) + trailing;
                cleanUrl = cleanUrl.slice(0, -1);
            }
        }

        return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer">${cleanUrl}</a>${trailing}`;
    });

    return escaped;
}

async function sendMessage() {
    const userMsg = input.value;
    if (!userMsg) return;

    // show user's message
    const userElem = document.createElement("div");
    userElem.className = "bubble user";
    userElem.textContent = userMsg;
    chatbox.appendChild(userElem);
    input.value = "";

    // disable further input
    input.disabled = true;
    button.disabled = true;

    // show loading animation
    const loadingElem = document.createElement("div");
    loadingElem.className = "bubble bot loading-bubble";
    loadingElem.innerHTML = `
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
    `;
    chatbox.appendChild(loadingElem);
    chatbox.scrollTop = chatbox.scrollHeight;


    try {
        // send message with session_id to FastAPI backend
        const res = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                message: userMsg,
                session_id: sessionId
            })
        });

        const data = await res.json();
        const botMsg = data.response;
        if (data.session_id) {
            sessionId = data.session_id;
            sessionStorage.setItem("chat_session_id", sessionId);
        }

        // remove loading animation
        loadingElem.remove();

        // show bot's message with formatted links
        const botElem = document.createElement("div");
        botElem.className = "bubble bot";
        botElem.innerHTML = formatMessage(botMsg);
        chatbox.appendChild(botElem);
    } catch (err) {
        const errElem = document.createElement("div");
        errElem.className = "bubble bot";
        errElem.textContent = "Error connecting to server. Please try again.";
        chatbox.appendChild(errElem);
    } finally {
        // re-enable input
        input.disabled = false;
        button.disabled = false;
        input.focus();
        chatbox.scrollTop = chatbox.scrollHeight;
    }
}

button.addEventListener("click", sendMessage);
input.addEventListener("keyup", (event) => {
    if (event.key === "Enter") {
        sendMessage();
    }
});