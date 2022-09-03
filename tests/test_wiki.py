import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "..",
                             "datasets"))
from wiki import WikiDatasetBuilder

TEST_TEXTS = [
    (
        "Castlewellan GAC (also known as St Malachy's GAC, or in Irish,  CLG Naomh Maolmhoig Caisleán a’ Mhuilinn) is a Gaelic Athletic Association Club in Castlewellan, County Down, Northern Ireland. "
        "The club promotes the Gaelic Games of Hurling, Football, Camogie and other cultural and social pursuits."
        "\n\nHistory\n"
        "The club was founded in 1905 and recently celebrated its centenary year in 2005."
        "\n\nAchievements\n"
        "Welcome to Castlewellan GAC, a club with over 113 years of history behind us. We have over 400 members involved in our various sporting, cultural, administrative and coaching activities. "
        "We have teams at all age groups in Gaelic Football (men and women), Hurling and Camogie. "
        "Men, women and children of all ages and all sections of the community - all are welcome in the club. "
        "We are always keen to see new members joining."
        "\n\n1920"
        " | Down Junior Football Champions (1st) 1924"
        " | Down Senior Football Champions (1st) 1927"
        " | Ulster Senior Handball Champions (doubles) 1927"
        " | East Division Senior Football Championship Winners 1934"
        " | Down Senior Football Champions (2nd) 1934"
        " | U14 Camogie – County Feile Shield Winners (1st)"
        "\n\nNotable players\n "
        "Pat Rice Member of the Down Senior team and won an Ulster & All-Ireland medal in 1960 & 1961. He also won a Down Senior Championship medals in 1965. "
        "\n Michael Cunningham former Down Goalkeeper"
        "\n\nSee also\n"
        "Down Senior Club Football Championship"
        "\nList of Gaelic Athletic Association clubs"
        "\nAn Riocht"
        "\nBredagh GAC"
        "\nClonduff GAC"
        "\nWarrenpoint GAA"
        "\n\nExternal links"
        "\nOfficial Castlewellan GAA Club website"
        "\nOfficial An Riocht GAA Club website"
        "\nOfficial Down County website"
        "\n\nGaelic Athletic Association clubs in County Down"
        "\nGaelic football clubs in County Down"
        "\nHurling clubs in County Down",
        [
            "Castlewellan GAC (also known as St Malachy's GAC, or in Irish, CLG Naomh Maolmhoig Caislen a Mhuilinn) is a Gaelic Athletic Association Club in Castlewellan, County Down, Northern Ireland.",
            'The club promotes the Gaelic Games of Hurling, Football, Camogie and other cultural and social pursuits.',
            'History',
            'The club was founded in 1905 and recently celebrated its centenary year in 2005.',
            'Achievements',
            'Welcome to Castlewellan GAC, a club with over 113 years of history behind us.',
            'We have over 400 members involved in our various sporting, cultural, administrative and coaching activities.',
            'We have teams at all age groups in Gaelic Football (men and women), Hurling and Camogie.',
            'Men, women and children of all ages and all sections of the community - all are welcome in the club.',
            'We are always keen to see new members joining.',
            '1920',
            'Down Junior Football Champions (1st) 1924',
            'Down Senior Football Champions (1st) 1927',
            'Ulster Senior Handball Champions (doubles) 1927',
            'East Division Senior Football Championship Winners 1934',
            'Down Senior Football Champions (2nd) 1934',
            'U14 Camogie County Feile Shield Winners (1st)',
            'Notable players',
            'Pat Rice Member of the Down Senior team and won an Ulster & All-Ireland medal in 1960 & 1961.',
            'He also won a Down Senior Championship medals in 1965.',
            'Michael Cunningham former Down Goalkeeper',
            'See also',
            'Down Senior Club Football Championship',
            'List of Gaelic Athletic Association clubs',
            'An Riocht',
            'Bredagh GAC',
            'Clonduff GAC',
            'Warrenpoint GAA',
            'External links',
            'Official Castlewellan GAA Club website',
            'Official An Riocht GAA Club website',
            'Official Down County website',
            'Gaelic Athletic Association clubs in County Down',
            'Gaelic football clubs in County Down',
            'Hurling clubs in County Down'
        ],
    ), (
        "Sharaf al-Din Khan b. Shams al-Din b. Sharaf Beg Bedlisi (Kurdish: شەرەفخانی بەدلیسی, Şerefxanê Bedlîsî; ; 25 February 1543 – ) was a Kurdish Emir of Bitlis. "
        "He was also a historian, writer and poet. "
        "He wrote exclusively in Persian. "
        "Born in the Qara Rud village, in central Iran, between Arak and Qom, at a young age he was sent to the Safavids\' court and obtained his education there."
        "\n\nHe is the author of Sharafnama, one of the most important works on medieval Kurdish history, written in 1597. "
        "He created a good picture of Kurdish life and Kurdish dynasties in the 16th century in his works. "
        "Outside Iran and Kurdish-speaking countries, Sharaf Khan Bidlisi has influenced Kurdish literature and societies through the translation of his works by other scholars."
        "\n\nHe was also a gifted artist and a well-educated man,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 excelling as much in mathematics and military strategy as he did in history.",
        [
            'Sharaf al-Din Khan b. Shams al-Din b. Sharaf Beg Bedlisi (Kurdish: , erefxan Bedls; ; 25 February 1543 ) was a Kurdish Emir of Bitlis.',
            'He was also a historian, writer and poet.',
            'He wrote exclusively in Persian.',
            "Born in the Qara Rud village, in central Iran, between Arak and Qom, at a young age he was sent to the Safavids' court and obtained his education there.",
            'He is the author of Sharafnama, one of the most important works on medieval Kurdish history, written in 1597.',
            'He created a good picture of Kurdish life and Kurdish dynasties in the 16th century in his works.',
            'Outside Iran and Kurdish-speaking countries, Sharaf Khan Bidlisi has influenced Kurdish literature and societies through the translation of his works by other scholars.',
            'He was also a gifted artist and a well-educated man, excelling as much in mathematics and military strategy as he did in history.'
        ]
    ), (
        'Kurosaki may refer to:'
        '\n\nPeople with the surname'
        '\n, Japanese former football player and manager'
        '\n, Japanese singer and songwriter'
        '\n, Japanese rower'
        '\n, Japanese actress, tarento, and fashion model'
        '\nRyan Kurosaki (born 1952), American former baseball player'
        '\n\nFictional characters'
        '\nIchigo Kurosaki, a character in Bleach '
        '\nIsshin Kurosaki, a character in Bleach'
        '\nMasaki Kurosaki, a character in Bleach'
        '\nKarin Kurosaki, a character in Bleach'
        '\nYuzu Kurosaki, a character in Bleach '
        '\nMea Kurosaki, a character in To Love-Ru Darkness '
        '\nTasuku Kurosaki, a character from the Dengeki Daisy manga series'
        '\nMiu Kurosaki, a character in The King of Fighters universe'
        '\nMiki Kurosaki, a character in the Digimon Data Squad'
        '\nSayoko Kurosaki and daughter Asami Kurosaki, characters in Mahoraba '
        '\nHisoka Kurosaki, a character in Descendants of Darkness '
        '\nShun Kurosaki and sister Ruri Kurosaki in Yu-Gi-Oh! Arc-V'
        '\n\nPlaces'
        '\nKurosaki, Niigata, a former town from Nishikanbara District in Niigata, Japan'
        '\n\nSee also'
        '\nKurosaki Station, a railway station in Japan'
        '\n\nJapanese-language surnames',
        [
            'Kurosaki may refer to:',
            'People with the surname',
            ', Japanese former football player and manager',
            ', Japanese singer and songwriter',
            ', Japanese rower',
            ', Japanese actress, tarento, and fashion model',
            'Ryan Kurosaki (born 1952), American former baseball player',
            'Fictional characters',
            'Ichigo Kurosaki, a character in Bleach',
            'Isshin Kurosaki, a character in Bleach',
            'Masaki Kurosaki, a character in Bleach',
            'Karin Kurosaki, a character in Bleach',
            'Yuzu Kurosaki, a character in Bleach',
            'Mea Kurosaki, a character in To Love-Ru Darkness',
            'Tasuku Kurosaki, a character from the Dengeki Daisy manga series',
            'Miu Kurosaki, a character in The King of Fighters universe',
            'Miki Kurosaki, a character in the Digimon Data Squad',
            'Sayoko Kurosaki and daughter Asami Kurosaki, characters in Mahoraba',
            'Hisoka Kurosaki, a character in Descendants of Darkness',
            'Shun Kurosaki and sister Ruri Kurosaki in Yu-Gi-Oh!',
            'Arc-V',
            'Places',
            'Kurosaki, Niigata, a former town from Nishikanbara District in Niigata, Japan',
            'See also',
            'Kurosaki Station, a railway station in Japan',
            'Japanese-language surnames'
        ]
    )
]

def test_match(text, ref_sents):
    test_sents = np.array(WikiDatasetBuilder.text_to_sents(text))
    ref_sents = np.array(ref_sents)

    len_test_sents = len(test_sents)
    len_ref_sents = len(ref_sents)
    if len_test_sents != len_ref_sents:
        print(f"EXPECTED {len_ref_sents} sentences vs. {len_test_sents} TEST sentences:", end="\n\n")
        for ref_sent in ref_sents:
            if ref_sent not in test_sents:
                print(f"<<", end="\t")
                print(ref_sent)
        for test_sent in test_sents:
            if test_sent not in ref_sents:
                print(f">>", end="\t")
                print(test_sent)
        return

    mismatches = np.where(test_sents!=ref_sents)[0]
    if len(mismatches) > 0:
        for i in mismatches:
            print(f"Mismatch in sentence {i}:", end="\n\n")
            print(f"EXPECTED: {ref_sents[i]}", end="\n\n")
            print(f"RESULT: {test_sents[i]}", end="\n\n")
            print()
    else:
        print("Result matches expected!")
