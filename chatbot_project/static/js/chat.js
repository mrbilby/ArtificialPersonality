
/* static/js/chat.js */
let messageContainer = document.getElementById('messageContainer');
let messageInput = document.getElementById('messageInput');

// static/js/chat.js
function addMessage(message, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    // Handle formatting
    const formattedMessage = message
        // Convert markdown headers
        .replace(/###\s*(.*)/g, '<h3>$1</h3>')
        // Convert numbered lists with proper spacing
        .replace(/(\d+\.\s+[^\n]+)/g, '<div class="list-item">$1</div>')
        // Convert emojis and special characters
        .replace(/([\u{1F300}-\u{1F6FF}])/gu, '<span class="emoji">$1</span>')
        // Convert **bold** text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Convert regular line breaks
        .replace(/\n/g, '<br>');
    
    content.innerHTML = formattedMessage;
    messageDiv.appendChild(content);
    messageContainer.appendChild(messageDiv);
    messageContainer.scrollTop = messageContainer.scrollHeight;
}

async function sendMessage(isBye = false) {
    const message = isBye ? 'bye' : messageInput.value.trim();
    if (!message) return;
    
    addMessage(message, true);
    messageInput.value = '';
    
    try {
        const response = await fetch('/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                personality: personalityName,
                is_bye: isBye
            })
        });
        
        const data = await response.json();
        if (data.error) {
            addMessage('Error: ' + data.error);
        } else {
            addMessage(data.response);
            if (data.terminated) {
                // Wait a moment for the goodbye message to be displayed
                setTimeout(() => {
                    window.location.href = '/';  // Redirect to personality selection
                }, 2000);
            }
        }
    } catch (error) {
        addMessage('Error: Could not send message');
    }
}

async function endChat() {
    if (confirm('Are you sure you want to end this chat? All memories will be saved.')) {
        await sendMessage(true);  // Send 'bye' command
    }
}

messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});