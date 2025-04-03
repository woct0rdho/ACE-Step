from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


random_genre_prompt = """randomly give me a short prompt that describes a music (with genre tag). less than 30 words
Here are some examples:
fusion jazz with synth, bass, drums, saxophone
Electronic, eerie, swing, dreamy, melodic, electro, sad, emotional
90s hip-hop, old school rap,  turntablism,  vinyl samples,  instrumental loop
"""


def random_genre():
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": random_genre_prompt}],
        max_tokens=30,
        temperature=0.7,
    )
    return completion.choices[0].message.content


optimize_genre_prompt = """optimize the following music descirption and make it more genre specific. less than 30 words
output examples:
fusion jazz with synth, bass, drums, saxophone
Electronic, eerie, swing, dreamy, melodic, electro, sad, emotional
90s hip-hop, old school rap,  turntablism,  vinyl samples,  instrumental loop

## input music descirption
"""


def optimize_genre(prompt):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": optimize_genre_prompt+prompt}],
        max_tokens=30,
        temperature=0.7,
    )
    return completion.choices[0].message.content
