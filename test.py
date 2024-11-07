import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

content = """
You are good at paraphrasing telugu text into telugu. 
For each sentence, give the Paraphrased_sentence: (more formal) in telugu,  a confidence score (How correct the text is).
"""

data = "వైకాపా నేత బోరుగడ్డ అనిల్‌కు పోలీసుల విందు భోజనం అంటూ సామాజిక మాధ్యమాల్లో వీడియో హల్‌చల్ చేస్తోంది. మంగళగిరి కోర్టులో హాజరు పరిచి రాజమండ్రి తరలిస్తుండగా గన్నవరం క్రాస్ రోడ్స్ రెస్టారెంట్‌లో అనిల్‌కు రాచ మర్యాదలు అంటూ విమర్శలు వ్యక్తమవుతున్నాయి. తెదేపా కార్యకర్తలు సెల్‌ఫోన్లో వీడియో చిత్రీకరిస్తుండగా.. పోలీసులు వాళ్ల ఫోన్ లాక్కుని వీడియో డిలీట్ చేశారు. సీసీ కెమెరా దృశ్యాలు సామాజిక మాధ్యమాల్లో చక్కర్లు కొడుతున్నాయి. విధుల్లో నిర్లక్ష్యంగా వ్యవహరించిన ఏడుగురు పోలీసులపై ఎస్పీ సతీశ్‌ కుమార్‌ సస్పెన్షన్‌ వేటు వేశారు."

client = Groq(
    api_key=os.environ['GROQ_API_KEY']
)

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": content},
        {"role": "user", "content": data}
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)