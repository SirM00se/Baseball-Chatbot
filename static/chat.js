async function sendMessage() {
    const input = document.getElementById("message");
    const chatbox = document.getElementById("chatbox");

    const userMsg = input.value;
    if (!userMsg) return;

    // show user's message
    chatbox.innerHTML += `<div class="bubble user">${userMsg}</div>`;
    input.value = "";

    // send message to FastAPI backend
    const res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: userMsg})
    });

    const data = await res.json();
    const botMsg = data.response;

    // show bot's message
    chatbox.innerHTML += `<div class="bubble bot">${botMsg}</div>`;
    chatbox.scrollTop = chatbox.scrollHeight;   // auto scroll to bottom
}