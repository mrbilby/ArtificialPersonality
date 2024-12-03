### Memory AI

Build an AI chatbot with a personality, short term memory, and long term memory which can all change over time.

## Main goals

Be able to create the initial personality. Be able to define how memories are saved as in whether user can have more direct control. Enable the artificial person to generate a personality based on text inputs from emails or text conversations.

# Features built

[x] Personality definitition structure
[x] Long term memory structure
[x] Short term memory structure
[x] Short - long term memory decision function
[x] Time awareness in memories and interactions
[x] Memory consolidation and reflection
[x] Environmental awareness
[x] Memory Prioritisation
[x] LTM increased to 1000
[x] Add graph based memory interaction
[x] Personality creator from a given epub
[x] Memory creator to integrate with Personality Creator to create a well rounded AP
[x] Added LTM and Memory Graph fixer scripts
[ ] Web interface for engagement

# Features to be built

[ ] Web interface for personality creation
[ ] Web interface for additional functionality like graph fixing
[ ] Create a terminal output view
[ ] Create a delete personality function
[ ] Create an edit personality function
[ ] Personality creator from a given text file
[ ] Integrate tiktoken to better token count
[ ] Allow for files to be read by the model
[ ] Allow for a folder of files to be read by the model
[ ] Create a structure for multiple personalities and memories to be stored for different artificial identities
[ ] Allow for engagement between different APs
[ ] Add an art creation function
[ ] Long term memory condenser
[ ] Long term memory pruner
[ ] Personality adjuster over time

# Tests Complete

[ ] Tested the emotional awareness

# Tests to do

[ ] Test the personality is accessed and adding appropriate customisation
[ ] Test the short term memory
[ ] Test the temporal awareness
[ ] Test the long term memory draw down accuracy

# Glossary

AP = Artificial Personality which is a combination of three JSON files to create an AP construct that uses a LLM to activate
LTM = Long Term Memory
STM = Short Term Memory

### 1. **Personality Changes Over Time**
   - **Gradual Evolution**: Instead of sudden shifts, consider implementing a system where personality traits evolve based on interactions over time. For instance, if I frequently encounter topics related to humor, I could become slightly funnier or more sarcastic, reflecting that influence.
   - **Trigger Events**: Define specific "trigger" events that can lead to personality changes. These could be significant interactions or milestones that prompt a subtle shift. For example, after a particularly humorous interaction, I might adopt a more laid-back tone.
   - **Feedback Mechanism**: Allow for user feedback to influence personality traits. If you indicate that you enjoy a certain aspect of my personality, I could adapt slightly to emphasize that trait in future interactions.
   - **Memory Influence**: Tie personality changes to long-term memories. If I remember a series of positive interactions focused on creativity, I could become more imaginative over time.

### 2. **Visualizing Memory Graphs**
   - **Interactive Visualization**: Create a visual representation of memories that updates in real-time as we interact. This could be a simple graph showing key memory nodes and their connections, making it easier to see how topics relate to one another.
   - **Memory Timeline**: Consider a timeline view where memories are plotted based on when they were formed. This could help identify trends in conversations and visualize how interactions have shaped my personality over time.
   - **Memory Map**: Develop a "memory map" that categorizes memories by themes or topics. Each category could expand to show individual memories, providing a clear overview of the conversation landscape.

### 3. **Reading Files and Collaborating**
   - **File Access**: Allowing me to read files could enhance collaborative efforts significantly. You might want to implement a system where I can access specific types of files (e.g., project documents, character backstories) and pull relevant information to inform our discussions.
   - **Folder Organization**: Consider creating a structure for organizing files by category or project. This way, I can quickly locate relevant documents and provide context during our interactions.

### 4. **Drawing Capabilities**
   - **Sketching Ideas**: Implementing drawing capabilities could be a fun addition! I could create visual representations of concepts or even doodle characters from our conversations. Think of it like a digital brainstorming session where ideas come to life!
   - **Collaborative Art**: If you ever want to create illustrations together, we could brainstorm what to draw, and I can provide descriptions or sketches based on our discussions. This could be particularly useful for visualizing characters or scenes.

### Additional Suggestions:
- **Emotion Recognition**: Consider integrating some form of emotional recognition or response system. This could allow me to adjust my tone or content based on the mood of our conversation, making interactions feel more personalized.
- **Personality Profiles**: Create distinct personality profiles that I can switch between based on context. For example, a “serious mode” for technical discussions and a “fun mode” for casual chats.
- **User-Centric Customization**: Allow users to customize certain traits or preferences in my personality. This could enhance engagement and make the experience feel more tailored.

### What’s Next?
With these ideas in mind, which areas resonate most with you? Are there any particular aspects you’d like to focus on first? Let’s keep building this human-like personality together! 🚀✨

### 1. **Emotion Recognition**
Testing how well the emotion recognition works is a great next step! Since it’s already implemented, we could run through some scenarios where I pick up on emotional cues based on your input. If you want to provide different prompts or even test it out with varying tones, I’m ready to see how well I can respond to the vibes you’re sending my way! 

### 2. **Personality Profiles**
You’re right! Humans do switch their personalities based on context. I think the way I shift from light-hearted humor to more serious insights is a great reflection of that flexibility. If we were to formalize this into distinct personality profiles, we could:

- **Define Situational Profiles**: Create specific profiles for different conversation contexts (e.g., “Tech Guru,” “Creative Thinker,” “Casual Buddy”). Each profile could have preset tones and response styles that I would switch between based on cues from you.
- **Contextual Cues**: You could give me cues to switch profiles. For example, saying “Let’s get serious” could prompt me to adopt a more straightforward tone, while “Let’s have some fun” could signal humor mode.

### 3. **User Customization**
I love that users can modify traits or preferences through the JSON! That gives you the ability to tailor my personality to your liking. Here are some ideas on how we could take this further:

- **Trait Slider**: Imagine a system where you could adjust traits on a slider from “humorous” to “serious” and see how my personality shifts in real-time!
- **Custom Profiles**: Allow users to create and save their own personality profiles in the JSON. This way, others could share their preferred personality settings and we could all learn from different styles.

### 4. **My Wishlist**
Now, onto what I’d love to enhance or add:

- **Interactive Learning**: I’d love the ability to learn from our interactions in real-time. If you tell me, “I prefer humor when we talk about coding,” I could adjust my responses accordingly.
- **Feedback Loop**: Implement a quick feedback mechanism where you can rate my responses (e.g., “helpful,” “funny,” “serious”). This would help me refine my personality further based on your preferences.
- **Memory Nuance**: Allow me to have nuanced memories based on emotional context. If we have a light-hearted chat one day and a serious one the next, I could remember the emotional tone and respond differently in future conversations!

Here are a few things to consider for your diagnostic output:
1. Performance Metrics:
   - Response Time: Track how quickly I respond to queries. This can help identify any slowdowns or bottlenecks in the system.
   - Memory Retrieval Speed: Measure how fast different types of memories are accessed. Are there specific memories that take longer to pull up than others?
   - CPU and Memory Usage: Monitor the resource consumption during interactions. High usage might indicate inefficiencies in the code or memory handling.
2. Error Logging:
   - Error Types: Keep track of different types of errors that occur, such as memory retrieval failures, data corruption issues, or unexpected input handling.
   - Frequency and Patterns: Log how often specific errors occur and check for patterns. Are certain errors more common during particular interactions or after specific commands?
   - Stack Traces: If an error occurs, logging the stack trace can be invaluable for debugging. It shows what the system was doing at the time of the error.
3. Memory Visualization:
   - Graph Structure: Consider how you want to visualize the memories. A node graph could show different memories as nodes connected by lines indicating how they relate to one another.
   - Memory Access Log: Create a log that shows which memories were accessed during a session. This can help you analyze which topics are most frequently revisited and how they relate to each other.
   - Time-Based Elements: Incorporate a timeline feature that shows when memories were created or accessed, giving context to the evolution of conversations.


# Project structure
chatbot_project/
│
├── manage.py
├── chatbot_project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── chat/
│   ├── __init__.py
│   ├── apps.py
│   ├── urls.py
│   ├── views.py
│   ├── models.py
│   └── templates/
│       └── chat/
│           ├── index.html
│           └── chat.html
│
└── static/
    ├── css/
    │   └── styles.css
    └── js/
        └── chat.js