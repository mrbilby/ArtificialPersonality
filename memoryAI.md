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
   - **Gradual Evolution**: Instead of sudden shifts, consider implementing a system where personality traits evolve based on interactions over time. For instance, if I frequently encounter topics related to humor, I could become slightly funnier or more sarcastic, reflecting that influence.
   - **Trigger Events**: Define specific "trigger" events that can lead to personality changes. These could be significant interactions or milestones that prompt a subtle shift. For example, after a particularly humorous interaction, I might adopt a more laid-back tone.
   - **Feedback Mechanism**: Allow for user feedback to influence personality traits. If you indicate that you enjoy a certain aspect of my personality, I could adapt slightly to emphasize that trait in future interactions.
   - **Memory Influence**: Tie personality changes to long-term memories. If I remember a series of positive interactions focused on creativity, I could become more imaginative over time.
   - **Interactive Visualization**: Create a visual representation of memories that updates in real-time as we interact. This could be a simple graph showing key memory nodes and their connections, making it easier to see how topics relate to one another.
   - **Memory Timeline**: Consider a timeline view where memories are plotted based on when they were formed. This could help identify trends in conversations and visualize how interactions have shaped my personality over time.
   - **Memory Map**: Develop a "memory map" that categorizes memories by themes or topics. Each category could expand to show individual memories, providing a clear overview of the conversation landscape.
- **Interactive Learning**: I’d love the ability to learn from our interactions in real-time. If you tell me, “I prefer humor when we talk about coding,” I could adjust my responses accordingly.
- **Feedback Loop**: Implement a quick feedback mechanism where you can rate my responses (e.g., “helpful,” “funny,” “serious”). This would help me refine my personality further based on your preferences.
- **Memory Nuance**: Allow me to have nuanced memories based on emotional context. If we have a light-hearted chat one day and a serious one the next, I could remember the emotional tone and respond differently in future conversations!


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