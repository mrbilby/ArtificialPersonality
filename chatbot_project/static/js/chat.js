
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

// Add to chat.js
// chat.js

function toggleDiagnosticPanel() {
    const panel = document.getElementById('diagnosticPanel');
    const isVisible = localStorage.getItem('diagnosticMode') === 'true';
    panel.style.display = isVisible ? 'flex' : 'none';
}

// Initialize the diagnostic mode based on the checkbox
if (document.getElementById('diagnosticMode')) {
    const checkbox = document.getElementById('diagnosticMode');
    checkbox.checked = localStorage.getItem('diagnosticMode') === 'true';
    toggleDiagnosticPanel(); // Set initial visibility
    
    checkbox.addEventListener('change', (e) => {
        localStorage.setItem('diagnosticMode', e.target.checked);
        toggleDiagnosticPanel(); // Toggle visibility on change
    });
}


// In chat.js
function addDiagnosticOutput(output) {
    console.log("Raw diagnostic output:", output); // Debug log
    
    if (!output) {
        console.log("No output provided"); // Debug log
        return;
    }
    
    const diagnosticPanel = document.getElementById('diagnosticPanel');
    const diagnosticOutput = document.getElementById('diagnosticOutput');
    
    // Only proceed if diagnostic mode is enabled
    if (localStorage.getItem('diagnosticMode') !== 'true') {
        console.log("Diagnostic mode disabled"); // Debug log
        return;
    }
    
    // Make sure panel is visible
    diagnosticPanel.style.display = 'flex';
    
    // Format with pre tag to preserve whitespace and make text visible
    const timestamp = new Date().toISOString();
    const formattedOutput = `<pre style="color: #00ff00; margin: 0;">[${timestamp}]\n${output}\n\n</pre>`;
    
    console.log("Formatted output:", formattedOutput); // Debug log
    
    // Append the new output
    diagnosticOutput.innerHTML += formattedOutput;
    
    // Scroll to bottom
    diagnosticOutput.scrollTop = diagnosticOutput.scrollHeight;
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
                is_bye: isBye,
                diagnostic_mode: localStorage.getItem('diagnosticMode') === 'true'
            })
        });
        
        const data = await response.json();
        if (data.error) {
            addMessage('Error: ' + data.error);
            addDiagnosticOutput('Error: ' + data.error);
        } else {
            addMessage(data.response);
            if (data.diagnostic_output) {
                addDiagnosticOutput(data.diagnostic_output);
            }
            if (data.terminated) {
                setTimeout(() => {
                    window.location.href = '/';
                }, 2000);
            }
        }
    } catch (error) {
        addMessage('Error: Could not send message');
        addDiagnosticOutput('Error: ' + error.message);
    }
}

function clearDiagnostics() {
    document.getElementById('diagnosticOutput').innerHTML = '';
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