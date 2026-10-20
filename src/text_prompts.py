"""
Multi-Prompt Vision-Language Text Ensemble for WS-VAD.
Implements fine-grained multi-prompt embedding generation for UCF-Crime and XD-Violence.
Averaging multi-prompt embeddings across descriptive variations eliminates single-prompt bias
and bridges the domain gap between natural text and CCTV surveillance footage.
"""

import os
import torch
import torch.nn.functional as F

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


UCF_CLASSES = [
    "normal", "abuse", "arrest", "arson", "assault", "burglary",
    "explosion", "fighting", "roadaccidents", "robbery", "shooting",
    "shoplifting", "stealing", "vandalism"
]

# Fine-grained multi-prompt ensemble (4 prompts per class)
UCF_MULTI_PROMPTS = {
    "normal": [
        "a surveillance video showing normal daily activity and routine life",
        "security camera footage of normal pedestrian traffic and peaceful surroundings",
        "cctv footage with no criminal activity, regular people walking normally",
        "a clear surveillance video of everyday ordinary public environment"
    ],
    "abuse": [
        "a surveillance video showing physical abuse and domestic violence",
        "security camera footage of a person hitting and abusing someone",
        "cctv footage of cruel violent mistreatment and physical abuse",
        "a surveillance camera capturing abusive behavior and bodily harm"
    ],
    "arrest": [
        "a surveillance video showing police officers arresting a suspect",
        "security camera footage of law enforcement handcuffing and detaining a person",
        "cctv recording of police arrest and suspect apprehension",
        "surveillance footage showing officers restraining and arresting an offender"
    ],
    "arson": [
        "a surveillance video showing arson and intentional fire",
        "security camera footage of someone setting fire to a building or property",
        "cctv video of burning flames, smoke, and criminal fire outbreak",
        "surveillance recording of arsonists deliberately igniting a blaze"
    ],
    "assault": [
        "a surveillance video showing violent physical assault",
        "security camera footage of someone being attacked and beaten up",
        "cctv footage of a violent physical aggression against a victim",
        "surveillance camera capturing violent punches and physical assault"
    ],
    "burglary": [
        "a surveillance video showing burglary and home break-in",
        "security camera footage of intruders breaking doors or windows to rob",
        "cctv recording of burglars illegally trespassing and ransacking property",
        "surveillance footage showing thieves breaking into a private premise"
    ],
    "explosion": [
        "a surveillance video showing a violent explosion and blast",
        "security camera footage of a bomb blast, shockwave, and debris",
        "cctv recording of exploding flames, heavy smoke, and sudden blast",
        "surveillance camera capturing a catastrophic building or vehicle explosion"
    ],
    "fighting": [
        "a surveillance video showing street fighting and physical brawl",
        "security camera footage of people punching, kicking, and fighting each other",
        "cctv footage of a violent brawl and aggressive brawl in public",
        "surveillance recording of multiple individuals engaged in a physical fistfight"
    ],
    "roadaccidents": [
        "a surveillance video showing a road accident and car crash",
        "security camera footage of a vehicle collision, crashing cars, or motorcycle crash",
        "cctv recording of a severe traffic accident and overturned vehicles",
        "traffic camera video of cars colliding violently on the road"
    ],
    "robbery": [
        "a surveillance video showing armed robbery and mugging",
        "security camera footage of robbers threatening victims with weapons for cash",
        "cctv footage of an armed store hold-up or violent street robbery",
        "surveillance camera capturing bandits looting a cashier at gunpoint"
    ],
    "shooting": [
        "a surveillance video showing gun shooting and firearm violence",
        "security camera footage of an active shooter firing a gun at people",
        "cctv recording of gunfire, muzzle flashes, and panicked crowds fleeing bullets",
        "surveillance footage of criminal shooting with pistols or rifles"
    ],
    "shoplifting": [
        "a surveillance video showing shoplifting in a store",
        "security camera footage of someone surreptitiously stealing merchandise in a shop",
        "cctv footage of a customer concealing items into pockets or bags without paying",
        "store surveillance camera catching a shoplifter taking goods"
    ],
    "stealing": [
        "a surveillance video showing theft and stealing personal property",
        "security camera footage of someone stealing a bicycle, purse, or unattended bag",
        "cctv recording of a thief snatching valuable belongings from a victim",
        "surveillance camera capturing petty theft and pickpocketing"
    ],
    "vandalism": [
        "a surveillance video showing vandalism and property destruction",
        "security camera footage of people smashing windows, graffitiing, and damaging property",
        "cctv footage of intentional malicious destruction of public facilities",
        "surveillance recording of rioters wrecking, smashing, and vandalizing street assets"
    ]
}


XD_CLASSES = [
    "normal", "fighting", "shooting", "riot", "abuse", "car accident", "explosion"
]

XD_MULTI_PROMPTS = {
    "normal": [
        "a video showing normal peaceful scenes with no violence",
        "peaceful daily life and calm environment without aggressive actions",
        "normal crowd movement and ordinary non-violent public behavior",
        "a calm video clip showing everyday peaceful activities"
    ],
    "fighting": [
        "a video showing violent fighting, fist brawl, and physical combat",
        "aggressive brawl where individuals are punching, kicking, and striking each other",
        "intense physical fight and brawl between aggressive people",
        "violent altercation with heavy blows, kicks, and physical combat"
    ],
    "shooting": [
        "a video showing gun shooting, firearm discharge, and bullet fire",
        "shootout with firearms, people firing guns and shooting weapons",
        "gun battle scene with rifles, pistols, gunshots, and muzzle flashes",
        "armed individuals shooting at targets with automatic firearms"
    ],
    "riot": [
        "a video showing public riot, mob chaos, and violent street protests",
        "chaotic civil unrest, protesters clashing with police, and street riots",
        "angry mob rioting, throwing objects, and causing street turmoil",
        "violent riot with widespread civil disorder and aggressive vandalism"
    ],
    "abuse": [
        "a video showing violence, abuse, and tormenting of a defenseless victim",
        "aggressive physical mistreatment and cruel violent abuse",
        "one person forcefully abusing, torturing, or violently beating another",
        "unprovoked violent assault and cruel bodily harm"
    ],
    "car accident": [
        "a video showing a car accident, vehicle collision, and road crash",
        "severe automotive crash with smashing vehicles and rolling cars",
        "violent collision of automobiles, trucks, or motorcycles",
        "traffic accident resulting in vehicle wreck and crash impact"
    ],
    "explosion": [
        "a video showing a massive explosion, detonation, and fiery blast",
        "violent bomb detonation with fireball, heavy smoke, and shockwave",
        "explosive blast destroying buildings or vehicles with intense fire",
        "catastrophic explosion throwing debris and fiery smoke into the air"
    ]
}

LABEL_MAP_UCF = {
    'Normal': 'normal',
    'Abuse': 'abuse',
    'Arrest': 'arrest',
    'Arson': 'arson',
    'Assault': 'assault',
    'Burglary': 'burglary',
    'Explosion': 'explosion',
    'Fighting': 'fighting',
    'RoadAccidents': 'roadaccidents',
    'Robbery': 'robbery',
    'Shooting': 'shooting',
    'Shoplifting': 'shoplifting',
    'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}

LABEL_MAP_XD = {
    'A': 'normal',
    'Normal': 'normal',
    'normal': 'normal',
    'B1': 'fighting',
    'B2': 'shooting',
    'B4': 'riot',
    'B5': 'abuse',
    'B6': 'car accident',
    'G': 'explosion'
}


def get_class_index(label_str, dataset='ucf'):
    """Maps label string from CSV to integer class index (0 is Normal)."""
    clean_str = str(label_str).strip()
    if dataset == 'ucf':
        mapped = LABEL_MAP_UCF.get(clean_str, 'normal')
        if mapped in UCF_CLASSES:
            return UCF_CLASSES.index(mapped)
        return 0
    else:
        mapped = LABEL_MAP_XD.get(clean_str, 'normal')
        if mapped in XD_CLASSES:
            return XD_CLASSES.index(mapped)
        return 0


def load_or_encode_text_features(device, dataset='ucf', clip_model_name="ViT-B/16"):
    """
    Computes ensemble multi-prompt CLIP text features of shape (C, 512).
    For each class, encodes 4 distinct prompts, averages their embeddings,
    and normalizes to the unit hypersphere.
    Caches the resulting tensor locally to avoid repeated CLIP forward passes.
    """
    cache_path = f"list/{dataset}_clip_text_features.pt"
    if os.path.exists(cache_path):
        features = torch.load(cache_path, map_location=device).float()
        return features

    classes = UCF_CLASSES if dataset == 'ucf' else XD_CLASSES
    multi_prompts = UCF_MULTI_PROMPTS if dataset == 'ucf' else XD_MULTI_PROMPTS

    if not HAS_CLIP:
        print("[Warning] openai-clip not installed. Generating normalized orthogonal text embeddings.")
        C = len(classes)
        features = torch.randn(C, 512, device=device).float()
        features = F.normalize(features, p=2, dim=-1)
        return features

    print(f"[TextPrompts] Encoding multi-prompt ensemble for {dataset.upper()} ({len(classes)} classes)...")
    clip_model, _ = clip.load(clip_model_name, device=device)
    clip_model.eval()

    class_embeddings = []
    with torch.no_grad():
        for cls_name in classes:
            prompts = multi_prompts[cls_name]
            tokens = clip.tokenize(prompts).to(device)
            # (num_prompts, 512) - cast to float32 immediately to avoid c10::Half mismatch
            prompt_feats = clip_model.encode_text(tokens).float()
            prompt_feats = F.normalize(prompt_feats, p=2, dim=-1)
            # Ensemble centroid: average across prompt variations
            ensemble_feat = torch.mean(prompt_feats, dim=0, keepdim=True)
            ensemble_feat = F.normalize(ensemble_feat, p=2, dim=-1)
            class_embeddings.append(ensemble_feat)

    # (C, 512) strictly float32
    text_features = torch.cat(class_embeddings, dim=0).float()

    # Free CLIP model from VRAM immediately
    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(text_features.cpu().float(), cache_path)
    print(f"[TextPrompts] Saved cached ensemble text features to {cache_path}")
    return text_features.to(device).float()
