import requests


'''
Complex Human Instruction from SANA:https://nvlabs.github.io/Sana/
'''
    # Here are examples of how to transform or refine prompts:
    #         - User Prompt: A cat sleeping -> refined:A serene scene of a fluffy, medium-sized cat peacefully sleeping. The cat has a soft, velvety coat with a mix of warm, earthy tones like caramel, chestnut, and hints of cream. Its paws are tucked under its body, and its tail curls gently around its legs. The cat's face is relaxed, with its eyes closed and whiskers slightly twitching. The background is a cozy, sunlit room with wooden floors and a large, plush cushion where the cat rests. The sunlight filters through sheer curtains, casting a gentle, golden glow over the scene, highlighting the cat's fur and adding a touch of warmth to the overall ambiance.
    #         - User Prompt: A busy city street -> refined: A bustling city street teeming with life and activity. The street is lined with tall, modern skyscrapers, their glass facades reflecting the vibrant colors of the setting sun. Neon signs flicker and glow, illuminating the sidewalk where pedestrians hurry by, some chatting on their phones, others engrossed in their newspapers. The street is wide, with multiple lanes of traffic flowing in both directions, filled with a mix of cars, buses, and taxis. The air is filled with the sounds of honking horns, chatter, and the distant hum of a subway train. Street vendors set up their stalls, selling everything from hot dogs to fresh flowers. A group of street performers entertain a small crowd with juggling and music. The scene is a mosaic of movement, color, and sound, capturing the essence of a lively urban environment.
            


def Complex_Human_Instruction(prompt_ori, model="gemma2:2b", max_token_num=240): #The refined prompt shoulde more than {max_token_num-100} words.
    prompt = f'''
            Given a user prompt, generate a very long "refined prompt" that provides detailed visual descriptions suitable for image generation. 
            Evaluate the level of detail in the user prompt:
            If the prompt is simple: Focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
            If the prompt is already detailed: Refine and enhance the existing details slightly without overcomplicating.
            
            Please generate only the refined prompt below and avoid including any additional commentary or evaluations:
                Examples:
                User Prompt: A sunset over the ocean.
                Refined Prompt: A breathtaking sunset over the vast ocean, where the sky is painted with hues of deep orange, pink, and purple. The sun, a glowing orb, dips below the horizon, casting long shadows across the calm, azure waters. Seagulls glide gracefully in the golden light, while gentle waves lap against the sandy shore, creating a serene and picturesque scene.

                User Prompt: A futuristic cityscape.
                Refined Prompt: A sprawling futuristic cityscape at night, illuminated by neon lights and towering skyscrapers. The buildings are sleek and angular, with glass facades reflecting the vibrant colors of the city. Flying vehicles zip through the air, leaving trails of light in their wake. The streets below are bustling with activity, filled with people and autonomous robots moving seamlessly through the urban environment.
                            
            Please generate only the refined prompt below and avoid including any additional commentary or evaluations:
            User Prompt:{prompt_ori}
            '''
 
            
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    result = response.json()
    try:
        return result["response"]
    except:
        print(result)


def Generate_Negative_Prompt(prompt_ori, model="gemma2:2b"):
    prompt = f'''
    Given a user prompt for image generation, create a detailed "Negative Prompt" that describes elements to avoid in the image. The negative prompt should be between 240 and 320 words. Follow these guidelines:

    1. Analyze the user prompt and identify potential undesired elements or qualities.
    2. Describe specific visual elements, styles, or themes to exclude, focusing on:
       - Unwanted objects, characters, or backgrounds
       - Inappropriate color schemes or lighting conditions
       - Undesired artistic styles or techniques
       - Potential technical issues or artifacts to avoid

    Examples:
    - User Prompt: "A serene beach at sunset"
    Negative Prompt: Crowded scenes, urban elements, or industrial objects. Stormy weather, dark clouds, or rough seas. Any modern technology, vehicles, or buildings. Unrealistic color palettes or overly saturated hues. Any text, logos, or watermarks. Poor composition, blurriness, or low resolution. Images with distorted proportions or anatomically incorrect elements.Any winter or cold weather imagery.

    - User Prompt: "A futuristic cityscape"
    Negative Prompt: Historical or ancient architectural styles. Natural landscapes, rural scenes, pastoral elements. Outdated technology or retro designs. Prevent dystopian or post-apocalyptic imagery.Overly cluttered or chaotic compositions. Any recognizable real-world landmarks or buildings. Images with a lack of technological elements or innovation. Dull or muted color schemes. Low-quality renders, pixelation, or obvious CGI artifacts.

    Now, generate only the negative prompt in one line for the following user prompt, without any additional commentary:
    User Prompt: {prompt_ori}
    '''
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    #neg_base_prompt = "worst quality, normal quality, low quality, low res, blurry, distortion, banner, extra digits, cropped, jpeg artifacts, username, error, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad proportions, bad quality, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, pixelated, grainy, color aberration, amputee, autograph, bad illustration, beyond the borders, blank background, body out of frame, boring background, branding, cut off, dismembered, disproportioned, distorted, draft, duplicated features, extra fingers, extra legs, fault, flaw, grains, hazy, identifying mark, improper scale, incorrect physiology, incorrect ratio, indistinct, kitsch, low resolution"
    response = requests.post(url, json=payload)
    result = response.json()

    return result["response"]#+neg_base_prompt



# print(Generate_Negative_Prompt("A cat sleeping"))