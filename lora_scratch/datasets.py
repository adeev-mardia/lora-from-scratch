"""
Synthetic-but-genuine text corpora for the two classification tasks used in
this project.

Both corpora are built from hand-written sentence templates and vocabulary
banks that are combined programmatically to produce a large number of
distinct, grammatical English sentences. This keeps the corpus large enough
for a measurable fine-tuning effect while staying fully offline and
reproducible (no external dataset download is involved).

Task A ("topic"): five-way news/article topic classification. This is the
task the base model is pretrained on.

Task B ("review"): three-way product-review sentiment classification, a
different label space and a different register of writing (first-person
product reviews rather than third-person news-style text). Fine-tuning on
Task B after pretraining on Task A is the genuine domain/task shift that
this project measures transfer on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Corpus:
    texts: list[str]
    labels: list[int]
    label_names: list[str]


# ---------------------------------------------------------------------------
# Task A: topic classification (pretraining task)
# ---------------------------------------------------------------------------

_TOPIC_LABELS = ["sports", "technology", "finance", "cooking", "politics"]

_TOPIC_SUBJECTS = {
    "sports": [
        "the home team", "the visiting squad", "the defending champions",
        "the rookie point guard", "the national team", "the veteran coach",
        "the league's top scorer", "the underdog club", "the relay team",
        "the marathon leader", "the goalkeeper", "the tennis finalist",
        "the Olympic committee", "the youth academy", "the rival franchise",
    ],
    "technology": [
        "the startup's engineering team", "the chip manufacturer",
        "the open-source maintainers", "the cloud provider",
        "the research lab", "the app's developers", "the hardware division",
        "the cybersecurity firm", "the robotics company",
        "the semiconductor foundry", "the search engine's algorithm team",
        "the data center operator", "the browser vendor",
        "the AI safety researchers", "the telecom carrier",
    ],
    "finance": [
        "the central bank", "the regional lender", "the hedge fund",
        "the finance ministry", "the stock exchange", "the credit rating agency",
        "the pension fund managers", "the venture capital firm",
        "the insurance underwriters", "the bond market traders",
        "the brokerage house", "the mortgage lender", "the commodities desk",
        "the sovereign wealth fund", "the audit committee",
    ],
    "cooking": [
        "the pastry chef", "the restaurant's head cook", "the home baker",
        "the food blogger", "the culinary school instructor",
        "the catering team", "the farmers' market vendor",
        "the cookbook author", "the line cook", "the sommelier",
        "the barista", "the butcher", "the food truck owner",
        "the grill master", "the recipe developer",
    ],
    "politics": [
        "the prime minister", "the opposition leader", "the city council",
        "the foreign ministry", "the parliamentary committee",
        "the sitting senator", "the regional governor", "the coalition partners",
        "the election commission", "the diplomatic envoy",
        "the cabinet spokesperson", "the trade negotiators",
        "the party's youth wing", "the sitting mayor", "the policy think tank",
    ],
}

_TOPIC_ACTIONS = {
    "sports": [
        "secured a narrow victory in the final minutes",
        "announced the signing of a new striker",
        "suffered a surprising defeat on home turf",
        "broke the tournament scoring record",
        "extended its unbeaten streak to ten matches",
        "appealed the referee's controversial decision",
        "unveiled a new training facility before the season",
        "clinched the division title with a late goal",
        "faced criticism after a disappointing performance",
        "celebrated a hard-fought comeback win",
        "qualified for the championship after a tense playoff",
        "rested its star players ahead of the derby",
    ],
    "technology": [
        "released a major update to its flagship product",
        "reported a data breach affecting millions of accounts",
        "unveiled a faster chip architecture at the conference",
        "open-sourced a key piece of its internal tooling",
        "patched a critical vulnerability after public disclosure",
        "announced layoffs amid a restructuring effort",
        "launched a new device aimed at enterprise customers",
        "faced regulatory scrutiny over data collection practices",
        "doubled its cloud computing capacity this quarter",
        "trained a new model on an expanded dataset",
        "shipped a long-delayed feature to all users",
        "formed a partnership with a leading hardware supplier",
    ],
    "finance": [
        "raised interest rates by a quarter point",
        "reported quarterly earnings above analyst expectations",
        "warned of slowing growth in the coming year",
        "downgraded the country's credit outlook",
        "announced a share buyback program",
        "tightened lending standards after rising defaults",
        "posted its largest quarterly loss in a decade",
        "revised its inflation forecast upward",
        "approved a merger between two mid-sized lenders",
        "cut its dividend amid falling revenue",
        "moved to stabilize the currency after a sharp selloff",
        "issued new bonds to fund infrastructure spending",
    ],
    "cooking": [
        "shared a new recipe for a seasonal dessert",
        "demonstrated a technique for searing fish perfectly",
        "revealed the secret ingredient in a signature sauce",
        "hosted a tasting event for the new seasonal menu",
        "explained how to properly knead sourdough dough",
        "recommended a wine pairing for the holiday dinner",
        "showcased a plant-based twist on a classic dish",
        "published a guide to fermenting vegetables at home",
        "taught a class on knife skills for beginners",
        "experimented with a fusion of two regional cuisines",
        "perfected a slow-roasting method for tougher cuts",
        "introduced a new line of artisanal bread",
    ],
    "politics": [
        "proposed a new bill to reform healthcare spending",
        "called for early elections amid coalition tensions",
        "signed a trade agreement with a neighboring country",
        "faced a vote of no confidence in parliament",
        "announced sweeping changes to immigration policy",
        "condemned the opposition's handling of the budget",
        "met with foreign leaders to discuss border security",
        "launched an inquiry into the procurement scandal",
        "pledged new funding for rural infrastructure",
        "negotiated a ceasefire after months of tension",
        "defended the decision to raise public sector wages",
        "unveiled a plan to cut carbon emissions by 2035",
    ],
}

_TOPIC_TAILS = {
    "sports": [
        "fans flooded social media with reactions.",
        "analysts called it one of the best performances this season.",
        "the result shifts the standings heading into the final stretch.",
        "the coach praised the team's resilience afterward.",
        "ticket sales for the next fixture rose sharply.",
        "pundits debated whether the decision was fair.",
    ],
    "technology": [
        "industry observers say the move could reshape the market.",
        "users reported mixed reactions on online forums.",
        "competitors are expected to respond within months.",
        "the announcement sent shares of rival firms lower.",
        "developers welcomed the change as long overdue.",
        "regulators said they would monitor the rollout closely.",
    ],
    "finance": [
        "markets reacted cautiously to the announcement.",
        "economists remain divided on the long-term impact.",
        "investors shifted toward safer assets in response.",
        "the move was widely anticipated by traders.",
        "borrowing costs are expected to rise as a result.",
        "small businesses said the change would strain their margins.",
    ],
    "cooking": [
        "the recipe quickly went viral online.",
        "home cooks praised how approachable the method was.",
        "the dish sold out within the first hour of service.",
        "critics called it a standout addition to the menu.",
        "readers asked for a step-by-step video to follow along.",
        "the technique has since been adopted by several local chefs.",
    ],
    "politics": [
        "the announcement drew mixed reactions from the public.",
        "opposition parties vowed to challenge the decision.",
        "analysts say the policy could face legal hurdles.",
        "citizens organized demonstrations in response.",
        "international observers welcomed the development.",
        "the decision is expected to dominate headlines for weeks.",
    ],
}


def generate_topic_corpus(per_class: int = 220, seed: int = 13) -> Corpus:
    rng = random.Random(seed)
    texts, labels = [], []
    for label_idx, name in enumerate(_TOPIC_LABELS):
        subjects = _TOPIC_SUBJECTS[name]
        actions = _TOPIC_ACTIONS[name]
        tails = _TOPIC_TAILS[name]
        seen = set()
        count = 0
        while count < per_class:
            subj = rng.choice(subjects)
            act = rng.choice(actions)
            tail = rng.choice(tails)
            sentence = f"{subj.capitalize()} {act}, and {tail}"
            if sentence in seen:
                continue
            seen.add(sentence)
            texts.append(sentence)
            labels.append(label_idx)
            count += 1
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return Corpus(list(texts), list(labels), _TOPIC_LABELS)


# ---------------------------------------------------------------------------
# Task B: product review sentiment classification (fine-tuning / domain-shift
# task). Different register (first person, product-review style) and a
# different label space (three sentiment classes) than Task A.
# ---------------------------------------------------------------------------

_REVIEW_LABELS = ["negative", "neutral", "positive"]

_REVIEW_PRODUCTS = [
    "wireless earbuds", "coffee maker", "running shoes", "laptop stand",
    "backpack", "desk lamp", "blender", "yoga mat", "phone case",
    "office chair", "bluetooth speaker", "water bottle", "electric kettle",
    "winter jacket", "gaming mouse", "air purifier", "notebook",
    "charging cable", "camping tent", "kitchen knife set",
]

_REVIEW_PHRASES = {
    "negative": [
        "stopped working after just a few days of normal use",
        "arrived damaged and the packaging offered no protection at all",
        "feels cheaply made and nowhere near worth the price",
        "broke down within a week and customer support was unhelpful",
        "the build quality is disappointing compared to the description",
        "started malfunctioning almost immediately out of the box",
        "is loud, uncomfortable, and far below what I expected",
        "had a defect straight out of the box and I had to return it",
        "fell apart after minimal use and I regret the purchase",
        "the battery barely lasts an hour despite the advertised claims",
    ],
    "neutral": [
        "works fine but doesn't really stand out from similar products",
        "does the job, though the instructions could be clearer",
        "is okay for the price but I wouldn't call it impressive",
        "has some nice touches but also a few rough edges",
        "performs as expected, nothing more and nothing less",
        "is decent overall, though shipping took longer than I liked",
        "gets the basic job done but lacks any standout features",
        "is a reasonable option if you don't need anything fancy",
        "works as described but the design feels a bit outdated",
        "is fine for casual use but not for anything demanding",
    ],
    "positive": [
        "exceeded my expectations and I use it every single day now",
        "is extremely well built and feels like a premium product",
        "arrived quickly and worked perfectly right out of the box",
        "has made a noticeable difference in my daily routine",
        "is by far the best purchase I've made in this category",
        "impressed everyone I showed it to with its build quality",
        "performs flawlessly even after months of heavy use",
        "was easy to set up and the customer service was excellent",
        "offers fantastic value and I would buy it again without hesitation",
        "feels durable, looks great, and works exactly as advertised",
    ],
}

_REVIEW_OPENERS = [
    "I bought this {p} last month and it",
    "This {p}", "Honestly, the {p} I ordered",
    "After two weeks with this {p}, it",
    "My new {p}",
    "I was skeptical at first, but this {p}",
    "Compared to my old one, this {p}",
    "My partner and I tried this {p}, and it",
]


def generate_review_corpus(per_class: int = 220, seed: int = 29) -> Corpus:
    rng = random.Random(seed)
    texts, labels = [], []
    for label_idx, name in enumerate(_REVIEW_LABELS):
        phrases = _REVIEW_PHRASES[name]
        seen = set()
        count = 0
        while count < per_class:
            product = rng.choice(_REVIEW_PRODUCTS)
            opener = rng.choice(_REVIEW_OPENERS).format(p=product)
            phrase = rng.choice(phrases)
            sentence = f"{opener} {phrase}."
            if sentence in seen:
                continue
            seen.add(sentence)
            texts.append(sentence)
            labels.append(label_idx)
            count += 1
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return Corpus(list(texts), list(labels), _REVIEW_LABELS)


def train_val_split(corpus: Corpus, val_frac: float = 0.2, seed: int = 7):
    rng = random.Random(seed)
    idx = list(range(len(corpus.texts)))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_idx = set(idx[:n_val])
    train = Corpus(
        [t for i, t in enumerate(corpus.texts) if i not in val_idx],
        [l for i, l in enumerate(corpus.labels) if i not in val_idx],
        corpus.label_names,
    )
    val = Corpus(
        [t for i, t in enumerate(corpus.texts) if i in val_idx],
        [l for i, l in enumerate(corpus.labels) if i in val_idx],
        corpus.label_names,
    )
    return train, val
