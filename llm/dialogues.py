TRAINING_DIALOGUES = [
    # Basic greetings and introductions
    "Hello there!", 
    "Greetings, traveler! How may I assist you today?",
    
    "What is this place?",
    "This is the ancient village of Mystwood, home to many secrets and mysteries.",
    
    "Tell me about your quest.",
    "I seek the legendary Crystal of Truth, hidden deep within these lands.",
    
    "Can you help me?",
    "Of course! I would be honored to aid you on your journey.",
    
    # Location and environment information
    "What dangers lurk here?",
    "Beware the shadow beasts that roam these woods after nightfall.",
    
    "Where should I go?",
    "The path to the east leads to the sacred temple, but be cautious.",
    
    # Items and inventory
    "Do you have any items?",
    "Yes, I have potions and scrolls that might aid your quest.",
    
    # Lore and secrets
    "Tell me a secret.",
    "They say the old well contains a passage to an ancient underground city.",
    
    "What's your story?",
    "I am the guardian of these lands, sworn to protect its ancient wisdom.",
    
    # Guidance and advice
    "I need guidance.",
    "Follow the stone markers, they will lead you to your destiny.",
    
    # Additional world-building dialogues
    "What's beyond the mountains?",
    "The Eternal Frost lands lie there, where few dare to venture.",
    
    "Tell me about the village.",
    "Our village was founded by ancient mages seeking refuge from the Great War.",
    
    "Who rules this land?",
    "The Council of Elders governs us with wisdom passed down through generations.",
    
    # Training dialogues for specific scenarios
    "I'm lost in the forest.",
    "Look for the glowing mushrooms, they always point toward the village.",
    
    "How can I become stronger?",
    "Seek out the Meditation Stone in the Valley of Whispers.",
    
    "What's that strange sound?",
    "Ah, those are the Wind Chimes of the Ancients, they sing when magic is near."
]

# You can add more specialized dialogue sets
COMBAT_DIALOGUES = [
    "Prepare for battle!",
    "Stand your ground, warrior!",
    
    "Show me your strength!",
    "My blade is ready for any challenge!",
    
    "Defend yourself!",
    "Your skills will be tested here!"
]

MERCHANT_DIALOGUES = [
    "What are you selling?",
    "I have rare artifacts from distant lands.",
    
    "How much for that potion?",
    "That'll be 50 gold pieces, a fair price for such quality.",
    
    "Can you offer a discount?",
    "For a fellow adventurer, I might consider it."
]

# Function to get combined dialogues
def get_all_dialogues():
    """Returns all dialogue sets combined"""
    return TRAINING_DIALOGUES + COMBAT_DIALOGUES + MERCHANT_DIALOGUES

# Function to get specific dialogue sets

def get_all_dialogues():
    """Returns all dialogue sets combined as list of dictionaries"""
    dialogues = []
    
    # Process TRAINING_DIALOGUES
    for i in range(0, len(TRAINING_DIALOGUES), 2):
        dialogues.append({
            'input': TRAINING_DIALOGUES[i],
            'response': TRAINING_DIALOGUES[i+1]
        })
    
    # Repeat for other dialogue lists
    for i in range(0, len(COMBAT_DIALOGUES), 2):
        dialogues.append({
            'input': COMBAT_DIALOGUES[i],
            'response': COMBAT_DIALOGUES[i+1]
        })
    
    for i in range(0, len(MERCHANT_DIALOGUES), 2):
        dialogues.append({
            'input': MERCHANT_DIALOGUES[i],
            'response': MERCHANT_DIALOGUES[i+1]
        })
    
    return dialogues

