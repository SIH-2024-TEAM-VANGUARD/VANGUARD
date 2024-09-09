document.addEventListener('DOMContentLoaded', function() {
    const messages = [];

    document.getElementById('send-btn').addEventListener('click', function() {
        const chatInput = document.getElementById('chat-input');
        const chatInputValue = chatInput.value;

        // Add the input text to the messages array
        messages.push({ sender: 'user', text: chatInputValue });

        // Send the entire conversation (including the new input) to the server
        fetch('/send_conversation_getinfo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ conversation: messages }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Conversation sent successfully:', data);

            // Update the chat body with the input text
            const chatBody = document.querySelector('.chat-body');
            const userMessage = document.createElement('div');
            userMessage.classList.add('message', 'user-message');
            userMessage.innerHTML = `<p>${chatInputValue}</p>`;
            chatBody.appendChild(userMessage);

            // Add the processed text to the messages array
            messages.push({ sender: 'bot', text: data.processed_text });

            // Add the processed text to the chat body
            const newMessage = document.createElement('div');
            newMessage.classList.add('message', 'bot-message');
            newMessage.innerHTML = `<p>${data.processed_text}</p>`;
            chatBody.appendChild(newMessage);

            // Play the generated speech
            const utterance = new SpeechSynthesisUtterance(data.processed_text);
            speechSynthesis.speak(utterance);

            // Clear the chat input
            chatInput.value = '';
        })
        .catch((error) => {
            console.error('Error sending conversation:', error);
        });
    });

    // Function to download the conversation
    document.getElementById('download-btn').addEventListener('click', function() {
        const conversationStr = JSON.stringify(messages, null, 2);
        const blob = new Blob([conversationStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'conversation.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });
});
