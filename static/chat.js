const input = document.getElementById("message");
const chatbox = document.getElementById("chatbox");
const button = document.getElementById("button");

// Persist or initialize a unique session ID in sessionStorage
let sessionId = sessionStorage.getItem("chat_session_id");
if (!sessionId) {
    sessionId = "session_" + Math.random().toString(36).substring(2, 11) + "_" + Date.now();
    sessionStorage.setItem("chat_session_id", sessionId);
}

async function sendMessage() {
    const userMsg = input.value;
    if (!userMsg) return;

    // show user's message
    chatbox.innerHTML += `<div class="bubble user">${userMsg}</div>`;
    input.value = "";
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

        // show bot's message
        chatbox.innerHTML += `<div class="bubble bot">${botMsg}</div>`;
    } catch (err) {
        chatbox.innerHTML += `<div class="bubble bot">Error connecting to server. Please try again.</div>`;
    }

    chatbox.scrollTop = chatbox.scrollHeight;   // auto scroll to bottom
}

button.addEventListener("click", sendMessage);
input.addEventListener("keyup", (event) => {
    if (event.key === "Enter") {
        sendMessage();
    }
});