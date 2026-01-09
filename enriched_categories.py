"""
Enriched category verbalizers for news classification.
Provides detailed descriptions for each category to improve classification accuracy.
"""

ENRICHED_CATEGORIES = {
    'ag_news': {
        'World': 'international affairs, foreign policy, geopolitics, global conflicts, diplomacy, international relations, cross-border events, worldwide developments',
        'Sports': 'athletic competitions, games, teams, players, championships, tournaments, leagues, sporting events, matches, scores, coaches, athletes',
        'Business': 'companies, corporations, markets, economy, finance, stocks, trading, mergers, acquisitions, earnings, business deals, industry news, economic policy',
        'Sci/Tech': 'technology, innovation, software, hardware, computers, internet, gadgets, scientific research, discoveries, digital products, tech companies, startups'
    },
    'bbc_news': {
        'business': 'companies, markets, economy, finance, stocks, trading, corporate news, mergers, acquisitions, earnings, business strategy, industry developments, economic indicators',
        'entertainment': 'movies, television, music, celebrities, Hollywood, awards shows, film releases, box office, streaming services, pop culture, media personalities',
        'politics': 'government, elections, politicians, legislation, policy, political parties, parliament, congress, campaigns, political debates, governance, public policy',
        'sport': 'athletic competitions, games, teams, players, championships, tournaments, leagues, matches, scores, football, cricket, tennis, Olympics, sporting events',
        'tech': 'technology, innovation, software, hardware, internet, gadgets, computing, digital products, tech companies, startups, telecommunications, cybersecurity, AI'
    },
    '20newsgroups': {
        'alt.atheism': 'atheism, religion debates, secular perspectives, non-belief, religious criticism, philosophy of religion, rationalism, skepticism',
        'comp.graphics': 'computer graphics, image processing, rendering, visualization, 3D modeling, graphics software, CAD, animation, graphic design',
        'comp.os.ms-windows.misc': 'Microsoft Windows operating system, Windows software, system configuration, Windows troubleshooting, MS-DOS, Windows applications',
        'comp.sys.ibm.pc.hardware': 'IBM PC hardware, computer components, motherboards, processors, memory, storage devices, PC building, hardware troubleshooting',
        'comp.sys.mac.hardware': 'Macintosh hardware, Apple computers, Mac components, system upgrades, Mac troubleshooting, peripherals',
        'comp.windows.x': 'X Window System, Unix graphics, X11, window managers, GUI programming, Unix desktop environments',
        'misc.forsale': 'items for sale, classified ads, buying and selling, marketplace listings, product advertisements, garage sales',
        'rec.autos': 'automobiles, cars, vehicles, automotive news, car reviews, maintenance, repairs, driving, car models',
        'rec.motorcycles': 'motorcycles, bikes, riding, motorcycle maintenance, motorcycle models, racing, motorcycle gear, biking culture',
        'rec.sport.baseball': 'baseball, MLB, teams, players, games, statistics, baseball history, leagues, ballparks',
        'rec.sport.hockey': 'hockey, NHL, ice hockey, teams, players, games, hockey statistics, leagues, rinks',
        'sci.crypt': 'cryptography, encryption, security, ciphers, cryptographic protocols, data protection, cryptanalysis',
        'sci.electronics': 'electronics, circuits, components, electronic devices, semiconductors, electrical engineering, circuit design',
        'sci.med': 'medicine, health, medical research, diseases, treatments, healthcare, doctors, medical conditions, symptoms',
        'sci.space': 'space exploration, astronomy, NASA, rockets, satellites, planets, space missions, astrophysics, spacecraft',
        'soc.religion.christian': 'Christianity, Christian faith, Bible, church, theology, Christian practices, religious discussions',
        'talk.politics.guns': 'gun control, firearms, Second Amendment, gun rights, gun legislation, weapons policy, gun ownership',
        'talk.politics.mideast': 'Middle East politics, Israel, Palestine, regional conflicts, Middle Eastern affairs, Arab-Israeli issues',
        'talk.politics.misc': 'political discussions, general politics, political opinions, governance debates, policy discussions',
        'talk.religion.misc': 'religion, faith, religious beliefs, spirituality, interfaith dialogue, religious practices, theological discussions'
    }
}


def get_enriched_categories_text(dataset_name, categories):
    """
    Format enriched categories for prompt insertion.
    
    Args:
        dataset_name: Name of the dataset (ag_news, bbc_news, 20newsgroups)
        categories: List of category names
    
    Returns:
        Formatted string with category descriptions
    """
    enriched = ENRICHED_CATEGORIES.get(dataset_name, {})
    lines = []
    for cat in categories:
        description = enriched.get(cat, '')
        if description:
            lines.append(f"- {cat}: {description}")
        else:
            lines.append(f"- {cat}")
    return '\n'.join(lines)
