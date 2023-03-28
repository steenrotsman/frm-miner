"""
This module provides a runtime comparison of FRM-Miner and Ostinato.

Data sets are supplied by the UCR archive as two .tsv files, a train set and a test set.
The train and test files are joined into one data set, for which the baseline is then mined

To replicate this experiment, add a copy of UCRArchive_2018 from https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/
Results are saved to the file runtime.csv and benchmark.R gives code to analyse the results.
"""
import pickle
from os.path import join
from itertools import product
from time import perf_counter
from collections import defaultdict

import numpy as np
import stumpy
import yfinance as yf
from tqdm import tqdm

from frm.miner import Miner

FILE = 'benchmark.csv'
FOLDER = 'UCRArchive_2018'
FILES = ['Mallat', 'OliveOil', 'ToeSegmentation1', 'InlineSkate', 'FaceAll']
ITER = 10
PARTITIONS = ['TRAIN', 'TEST']

# Parameters for FRM-Miner
MINSUP = [0.3, 0.5, 0.7, 0.9]
SEGLEN = [4, 8, 16]
ALPHABET = [5, 7, 9]
MIN_LEN = [3]
MAX_OVERLAP = [0.7, 0.8, 0.9]

# Parameters for baseline algorithms
LENGTH = [25, 50, 100]


def main():
    # Get already calculated combinations from file
    seen = defaultdict(list)
    with open(FILE) as fp:
        for row in fp.readlines():
            fields = row.split(',')
            seen[fields[0]].append(fields[1])

    # Calculate and save run times to file
    with open(FILE, 'a') as fp:
        for name, data in get_data():
            benchmark_mm(data, name, fp, seen[name])
            benchmark_ostinato(data, name, fp, seen[name])

        name = 'Stocks'
        benchmark_mm(get_stocks(), name, fp, seen[name])
        benchmark_ostinato([np.array(stock, dtype=np.float64) for stock in get_stocks()], name, fp, seen[name])


def get_data():
    for fn in FILES:
        data = []
        for part in PARTITIONS:
            with open(join(FOLDER, fn, f'{fn}_{part}.tsv')) as f:
                for row in f:
                    # Split data on tabs
                    data.append(row.strip('\n').split('\t')[1:])

        # Parse data to floats
        data = [[float(x) for x in row] for row in data]

        yield fn, data


def benchmark_mm(data, name, fp, seen):
    for i, minsup, s, a, l, o in tqdm(list(product(range(ITER), MINSUP, SEGLEN, ALPHABET, MIN_LEN, MAX_OVERLAP))):
        if (combination := f'mm_{minsup}_{s}_{a}_{l}_{o}_{i}') in seen:
            continue

        start = perf_counter()
        mm = Miner(data, minsup, s, a, l, o)
        mm.mine_motifs()
        end = perf_counter()

        fp.write(f'{name},{combination},{end-start}\n')


def benchmark_ostinato(data, name, fp, seen):
    for i, m in tqdm(list(product(range(ITER), LENGTH))):
        if (combination := f'ostinato_{m}_{i}') in seen:
            continue

        start = perf_counter()
        stumpy.ostinato(data, m)
        end = perf_counter()

        fp.write(f'{name},{combination},{end-start}\n')


def get_stocks():
    try:
        with open('stocks.pkl', 'rb') as fp:
            volumes = pickle.load(fp)
    except FileNotFoundError:
        # https://stockanalysis.com/stocks/
        tickers = ['A', 'AA', 'AAC', 'AACG', 'AACI', 'AADI', 'AAIC', 'AAL', 'AAMC', 'AAME', 'AAN', 'AAOI', 'AAON',
                   'AAP', 'AAPL', 'AAT', 'AAU', 'AB', 'ABB', 'ABBV', 'ABC', 'ABCB', 'ABCL', 'ABCM', 'ABEO', 'ABEV',
                   'ABG', 'ABIO', 'ABM', 'ABNB', 'ABOS', 'ABR', 'ABSI', 'ABST', 'ABT', 'ABUS', 'ABVC', 'AC', 'ACA',
                   'ACAB', 'ACAC', 'ACAD', 'ACAH', 'ACAQ', 'ACAX', 'ACB', 'ACBA', 'ACCD', 'ACCO', 'ACDC', 'ACEL',
                   'ACER', 'ACET', 'ACGL', 'ACGN', 'ACHC', 'ACHL', 'ACHR', 'ACHV', 'ACI', 'ACIU', 'ACIW', 'ACLS',
                   'ACLX', 'ACM', 'ACMR', 'ACN', 'ACNB', 'ACNT', 'ACON', 'ACOR', 'ACR', 'ACRE', 'ACRO', 'ACRS', 'ACRV',
                   'ACRX', 'ACST', 'ACT', 'ACTG', 'ACU', 'ACVA', 'ACXP', 'ADAG', 'ADAL', 'ADAP', 'ADBE', 'ADC', 'ADCT',
                   'ADD', 'ADEA', 'ADER', 'ADES', 'ADEX', 'ADI', 'ADIL', 'ADM', 'ADMA', 'ADMP', 'ADN', 'ADNT', 'ADOC',
                   'ADP', 'ADPT', 'ADRT', 'ADSE', 'ADSK', 'ADT', 'ADTH', 'ADTN', 'ADTX', 'ADUS', 'ADV', 'ADVM', 'ADXN',
                   'AE', 'AEAE', 'AEE', 'AEG', 'AEHL', 'AEHR', 'AEI', 'AEIS', 'AEL', 'AEM', 'AEMD', 'AENZ', 'AEO',
                   'AEP', 'AER', 'AES', 'AESI', 'AEVA', 'AEY', 'AEYE', 'AEZS', 'AFAR', 'AFBI', 'AFCG', 'AFG', 'AFIB',
                   'AFL', 'AFMD', 'AFRI', 'AFRM', 'AFTR', 'AFYA', 'AG', 'AGAC', 'AGAE', 'AGBA', 'AGCO', 'AGE', 'AGEN',
                   'AGFS', 'AGFY', 'AGI', 'AGIL', 'AGIO', 'AGL', 'AGLE', 'AGM', 'AGM.A', 'AGMH', 'AGNC', 'AGO', 'AGR',
                   'AGRI', 'AGRO', 'AGRX', 'AGS', 'AGTI', 'AGX', 'AGYS', 'AHCO', 'AHG', 'AHH', 'AHI', 'AHRN', 'AHT',
                   'AI', 'AIB', 'AIG', 'AIH', 'AIHS', 'AIM', 'AIMAU', 'AIMC', 'AIMD', 'AIN', 'AINC', 'AIP', 'AIR',
                   'AIRC', 'AIRG', 'AIRI', 'AIRS', 'AIRT', 'AIT', 'AIU', 'AIV', 'AIXI', 'AIZ', 'AJG', 'AJRD', 'AJX',
                   'AKA', 'AKAM', 'AKAN', 'AKBA', 'AKLI', 'AKO.A', 'AKO.B', 'AKR', 'AKRO', 'AKTS', 'AKTX', 'AKU',
                   'AKYA', 'AL', 'ALAR', 'ALB', 'ALBT', 'ALC', 'ALCC', 'ALCO', 'ALDX', 'ALE', 'ALEC', 'ALEX', 'ALG',
                   'ALGM', 'ALGN', 'ALGS', 'ALGT', 'ALHC', 'ALIM', 'ALIT', 'ALK', 'ALKS', 'ALKT', 'ALL', 'ALLE', 'ALLG',
                   'ALLK', 'ALLO', 'ALLR', 'ALLT', 'ALLY', 'ALNY', 'ALOR', 'ALOT', 'ALPA', 'ALPN', 'ALPP', 'ALPS',
                   'ALRM', 'ALRN', 'ALRS', 'ALSA', 'ALSN', 'ALT', 'ALTG', 'ALTI', 'ALTO', 'ALTR', 'ALTU', 'ALV', 'ALVO',
                   'ALVR', 'ALX', 'ALXO', 'ALYA', 'ALZN', 'AM', 'AMAL', 'AMAM', 'AMAO', 'AMAT', 'AMBA', 'AMBC', 'AMBI',
                   'AMBO', 'AMBP', 'AMC', 'AMCR', 'AMCX', 'AMD', 'AME', 'AMED', 'AMEH', 'AMG', 'AMGN', 'AMH', 'AMK',
                   'AMKR', 'AMLI', 'AMLX', 'AMN', 'AMNB', 'AMOT', 'AMP', 'AMPE', 'AMPG', 'AMPH', 'AMPL', 'AMPS', 'AMPX',
                   'AMPY', 'AMR', 'AMRC', 'AMRK', 'AMRN', 'AMRS', 'AMRX', 'AMS', 'AMSC', 'AMSF', 'AMST', 'AMSWA', 'AMT',
                   'AMTB', 'AMTD', 'AMTI', 'AMTX', 'AMV', 'AMWD', 'AMWL', 'AMX', 'AMYT', 'AMZN', 'AN', 'ANAB', 'ANDE',
                   'ANEB', 'ANET', 'ANF', 'ANGH', 'ANGI', 'ANGN', 'ANGO', 'ANIK', 'ANIP', 'ANIX', 'ANNX', 'ANPC',
                   'ANSS', 'ANTE', 'ANTX', 'ANVS', 'ANY', 'ANZU', 'AOGO', 'AOMR', 'AON', 'AORT', 'AOS', 'AOSL', 'AOUT',
                   'AP', 'APA', 'APAC', 'APAM', 'APCA', 'APCX', 'APD', 'APDN', 'APEI', 'APEN', 'APG', 'APGB', 'APGN',
                   'APH', 'API', 'APLD', 'APLE', 'APLS', 'APLT', 'APM', 'APMI', 'APO', 'APOG', 'APP', 'APPF', 'APPH',
                   'APPN', 'APPS', 'APRE', 'APRN', 'APT', 'APTM', 'APTO', 'APTV', 'APTX', 'APVO', 'APWC', 'APXI',
                   'APYX', 'AQB', 'AQMS', 'AQN', 'AQST', 'AQU', 'AQUA', 'AR', 'ARAV', 'ARAY', 'ARBE', 'ARBG', 'ARBK',
                   'ARC', 'ARCB', 'ARCC', 'ARCE', 'ARCH', 'ARCO', 'ARCT', 'ARDS', 'ARDX', 'ARE', 'AREB', 'AREC', 'AREN',
                   'ARES', 'ARGO', 'ARGX', 'ARHS', 'ARI', 'ARIS', 'ARIZ', 'ARKO', 'ARKR', 'ARL', 'ARLO', 'ARLP', 'ARMK',
                   'ARMP', 'ARNC', 'AROC', 'AROW', 'ARQQ', 'ARQT', 'ARR', 'ARRW', 'ARRY', 'ARTE', 'ARTL', 'ARTNA',
                   'ARTW', 'ARVL', 'ARVN', 'ARW', 'ARWR', 'ARYD', 'ARYE', 'ASA', 'ASAI', 'ASAN', 'ASB', 'ASC', 'ASCA',
                   'ASCB', 'ASGN', 'ASH', 'ASIX', 'ASLE', 'ASLN', 'ASM', 'ASMB', 'ASML', 'ASND', 'ASNS', 'ASO', 'ASPA',
                   'ASPI', 'ASPN', 'ASPS', 'ASPU', 'ASR', 'ASRT', 'ASRV', 'ASST', 'ASTC', 'ASTE', 'ASTI', 'ASTL',
                   'ASTR', 'ASTS', 'ASUR', 'ASX', 'ASXC', 'ASYS', 'ATAI', 'ATAK', 'ATAQ', 'ATAT', 'ATCO', 'ATCX',
                   'ATEC', 'ATEK', 'ATEN', 'ATER', 'ATEX', 'ATGE', 'ATHA', 'ATHE', 'ATHM', 'ATHX', 'ATI', 'ATIF',
                   'ATIP', 'ATKR', 'ATLC', 'ATLO', 'ATLX', 'ATMC', 'ATMV', 'ATNF', 'ATNI', 'ATNM', 'ATNX', 'ATO',
                   'ATOM', 'ATOS', 'ATR', 'ATRA', 'ATRC', 'ATRI', 'ATRO', 'ATSG', 'ATTO', 'ATUS', 'ATVI', 'ATXG',
                   'ATXI', 'ATXS', 'ATY', 'AU', 'AUB', 'AUBN', 'AUD', 'AUDC', 'AUGX', 'AUID', 'AULT', 'AUMN', 'AUPH',
                   'AUR', 'AURA', 'AURC', 'AUST', 'AUTL', 'AUUD', 'AUVI', 'AUY', 'AVA', 'AVAC', 'AVAH', 'AVAL', 'AVAV',
                   'AVB', 'AVD', 'AVDL', 'AVDX', 'AVGO', 'AVGR', 'AVHI', 'AVID', 'AVIR', 'AVNS', 'AVNT', 'AVNW', 'AVO',
                   'AVPT', 'AVRO', 'AVT', 'AVTA', 'AVTE', 'AVTR', 'AVTX', 'AVXL', 'AVY', 'AWH', 'AWI', 'AWIN', 'AWK',
                   'AWR', 'AWRE', 'AWX', 'AX', 'AXAC', 'AXDX', 'AXGN', 'AXL', 'AXLA', 'AXNX', 'AXON', 'AXP', 'AXR',
                   'AXS', 'AXSM', 'AXTA', 'AXTI', 'AY', 'AYI', 'AYRO', 'AYTU', 'AYX', 'AZ', 'AZEK', 'AZN', 'AZO',
                   'AZPN', 'AZRE', 'AZTA', 'AZUL', 'AZYO', 'AZZ', 'B', 'BA', 'BABA', 'BAC', 'BACA', 'BACK', 'BAER',
                   'BAFN', 'BAH', 'BAK', 'BALL', 'BALY', 'BAM', 'BANC', 'BAND', 'BANF', 'BANL', 'BANR', 'BANX', 'BAOS',
                   'BAP', 'BARK', 'BASE', 'BATL', 'BAX', 'BB', 'BBAI', 'BBAR', 'BBBY', 'BBCP', 'BBD', 'BBDC', 'BBDO',
                   'BBGI', 'BBIG', 'BBIO', 'BBLG', 'BBLN', 'BBSI', 'BBU', 'BBUC', 'BBVA', 'BBW', 'BBWI', 'BBY', 'BC',
                   'BCAB', 'BCAN', 'BCBP', 'BCC', 'BCDA', 'BCE', 'BCEL', 'BCH', 'BCLI', 'BCML', 'BCO', 'BCOV', 'BCOW',
                   'BCPC', 'BCRX', 'BCS', 'BCSA', 'BCSF', 'BCTX', 'BCYC', 'BDC', 'BDL', 'BDN', 'BDSX', 'BDTX', 'BDX',
                   'BE', 'BEAM', 'BEAT', 'BECN', 'BEDU', 'BEEM', 'BEKE', 'BELFA', 'BELFB', 'BEN', 'BEP', 'BEPC', 'BERY',
                   'BEST', 'BF.A', 'BF.B', 'BFAC', 'BFAM', 'BFC', 'BFH', 'BFI', 'BFIN', 'BFLY', 'BFRG', 'BFRI', 'BFS',
                   'BFST', 'BG', 'BGCP', 'BGFV', 'BGI', 'BGNE', 'BGRY', 'BGS', 'BGSF', 'BGXX', 'BH', 'BH.A', 'BHAC',
                   'BHAT', 'BHB', 'BHC', 'BHE', 'BHF', 'BHG', 'BHIL', 'BHLB', 'BHM', 'BHP', 'BHR', 'BHVN', 'BIAF',
                   'BIDU', 'BIG', 'BIGC', 'BIIB', 'BILI', 'BILL', 'BIMI', 'BIO', 'BIOC', 'BIOL', 'BIOR', 'BIOS', 'BIOX',
                   'BIP', 'BIPC', 'BIRD', 'BITE', 'BITF', 'BIVI', 'BJ', 'BJDX', 'BJRI', 'BK', 'BKCC', 'BKD', 'BKE',
                   'BKH', 'BKI', 'BKKT', 'BKNG', 'BKR', 'BKSC', 'BKSY', 'BKTI', 'BKU', 'BKYI', 'BL', 'BLAC', 'BLBD',
                   'BLBX', 'BLCM', 'BLCO', 'BLD', 'BLDE', 'BLDP', 'BLDR', 'BLEU', 'BLFS', 'BLFY', 'BLIN', 'BLK', 'BLKB',
                   'BLMN', 'BLND', 'BLNG', 'BLNK', 'BLPH', 'BLRX', 'BLTE', 'BLU', 'BLUA', 'BLUE', 'BLX', 'BLZE', 'BMA',
                   'BMAC', 'BMAQ', 'BMBL', 'BMEA', 'BMI', 'BMO', 'BMR', 'BMRA', 'BMRC', 'BMRN', 'BMTX', 'BMY', 'BN',
                   'BNED', 'BNGO', 'BNIX', 'BNL', 'BNMV', 'BNNR', 'BNOX', 'BNR', 'BNRE', 'BNRG', 'BNS', 'BNSO', 'BNTC',
                   'BNTX', 'BOAC', 'BOC', 'BOCN', 'BODY', 'BOH', 'BOKF', 'BOLT', 'BON', 'BOOM', 'BOOT', 'BORR', 'BOSC',
                   'BOTJ', 'BOWL', 'BOX', 'BOXD', 'BOXL', 'BP', 'BPAC', 'BPMC', 'BPOP', 'BPRN', 'BPT', 'BPTH', 'BPTS',
                   'BQ', 'BR', 'BRAC', 'BRAG', 'BRBR', 'BRBS', 'BRC', 'BRCC', 'BRD', 'BRDG', 'BRDS', 'BREA', 'BREZ',
                   'BRFH', 'BRFS', 'BRID', 'BRIV', 'BRK.A', 'BRK.B', 'BRKH', 'BRKL', 'BRKR', 'BRLI', 'BRLT', 'BRMK',
                   'BRN', 'BRO', 'BROG', 'BROS', 'BRP', 'BRQS', 'BRSH', 'BRSP', 'BRT', 'BRTX', 'BRX', 'BRY', 'BRZE',
                   'BSAC', 'BSAQ', 'BSBK', 'BSBR', 'BSET', 'BSFC', 'BSGA', 'BSGM', 'BSIG', 'BSM', 'BSMX', 'BSQR',
                   'BSRR', 'BSVN', 'BSX', 'BSY', 'BTAI', 'BTB', 'BTBD', 'BTBT', 'BTCM', 'BTCS', 'BTCY', 'BTE', 'BTG',
                   'BTI', 'BTMD', 'BTOG', 'BTTR', 'BTTX', 'BTU', 'BTWN', 'BUD', 'BUR', 'BURL', 'BURU', 'BUSE', 'BV',
                   'BVH', 'BVN', 'BVS', 'BVXV', 'BW', 'BWA', 'BWAC', 'BWAQ', 'BWAY', 'BWB', 'BWC', 'BWEN', 'BWFG',
                   'BWMN', 'BWMX', 'BWV', 'BWXT', 'BX', 'BXC', 'BXMT', 'BXP', 'BXRX', 'BY', 'BYD', 'BYFC', 'BYN',
                   'BYND', 'BYNO', 'BYRN', 'BYSI', 'BYTS', 'BZ', 'BZFD', 'BZH', 'BZUN', 'C', 'CAAP', 'CAAS', 'CABA',
                   'CABO', 'CAC', 'CACC', 'CACI', 'CACO', 'CADE', 'CADL', 'CAE', 'CAG', 'CAH', 'CAKE', 'CAL', 'CALB',
                   'CALC', 'CALM', 'CALT', 'CALX', 'CAMP', 'CAMT', 'CAN', 'CANF', 'CANG', 'CANO', 'CAPL', 'CAPR', 'CAR',
                   'CARA', 'CARE', 'CARG', 'CARM', 'CARR', 'CARS', 'CARV', 'CASA', 'CASH', 'CASI', 'CASS', 'CASY',
                   'CAT', 'CATC', 'CATO', 'CATX', 'CATY', 'CB', 'CBAN', 'CBAT', 'CBAY', 'CBD', 'CBFV', 'CBIO', 'CBL',
                   'CBNK', 'CBOE', 'CBRE', 'CBRG', 'CBRL', 'CBSH', 'CBT', 'CBU', 'CBZ', 'CC', 'CCAI', 'CCAP', 'CCB',
                   'CCBG', 'CCCC', 'CCCS', 'CCEL', 'CCEP', 'CCF', 'CCI', 'CCJ', 'CCK', 'CCL', 'CCLD', 'CCLP', 'CCM',
                   'CCNE', 'CCO', 'CCOI', 'CCRD', 'CCRN', 'CCS', 'CCSI', 'CCTS', 'CCU', 'CCV', 'CCVI', 'CD', 'CDAK',
                   'CDAQ', 'CDAY', 'CDE', 'CDIO', 'CDLX', 'CDMO', 'CDNA', 'CDNS', 'CDRE', 'CDRO', 'CDTX', 'CDW', 'CDXC',
                   'CDXS', 'CDZI', 'CE', 'CEAD', 'CECO', 'CEG', 'CEI', 'CEIX', 'CELC', 'CELH', 'CELL', 'CELU', 'CELZ',
                   'CEMI', 'CENN', 'CENT', 'CENTA', 'CENX', 'CEPU', 'CEQP', 'CERE', 'CERS', 'CERT', 'CET', 'CETU',
                   'CETX', 'CEVA', 'CF', 'CFB', 'CFBK', 'CFFE', 'CFFI', 'CFFN', 'CFFS', 'CFG', 'CFIV', 'CFLT', 'CFMS',
                   'CFR', 'CFRX', 'CFSB', 'CG', 'CGA', 'CGAU', 'CGBD', 'CGC', 'CGEM', 'CGEN', 'CGNT', 'CGNX', 'CGRN',
                   'CGTX', 'CHAA', 'CHCI', 'CHCO', 'CHCT', 'CHD', 'CHDN', 'CHE', 'CHEA', 'CHEF', 'CHEK', 'CHGG', 'CHH',
                   'CHK', 'CHKP', 'CHMG', 'CHMI', 'CHNR', 'CHPT', 'CHRA', 'CHRD', 'CHRS', 'CHRW', 'CHS', 'CHT', 'CHTR',
                   'CHUY', 'CHWY', 'CHX', 'CI', 'CIA', 'CIB', 'CIDM', 'CIEN', 'CIFR', 'CIG', 'CIGI', 'CIH', 'CIIG',
                   'CIM', 'CINF', 'CING', 'CINT', 'CIO', 'CION', 'CIR', 'CISO', 'CITE', 'CIVB', 'CIVI', 'CIX', 'CIZN',
                   'CJJD', 'CKPT', 'CKX', 'CL', 'CLAR', 'CLAY', 'CLB', 'CLBK', 'CLBR', 'CLBT', 'CLCO', 'CLDT', 'CLDX',
                   'CLEU', 'CLF', 'CLFD', 'CLGN', 'CLH', 'CLIN', 'CLIR', 'CLLS', 'CLMB', 'CLMT', 'CLNE', 'CLNN', 'CLOE',
                   'CLOV', 'CLPR', 'CLPS', 'CLPT', 'CLRB', 'CLRC', 'CLRO', 'CLS', 'CLSD', 'CLSK', 'CLST', 'CLVR',
                   'CLVT', 'CLW', 'CLWT', 'CLX', 'CLXT', 'CM', 'CMA', 'CMAX', 'CMBM', 'CMC', 'CMCA', 'CMCL', 'CMCM',
                   'CMCO', 'CMCSA', 'CMCT', 'CME', 'CMG', 'CMI', 'CMLS', 'CMMB', 'CMND', 'CMP', 'CMPO', 'CMPR', 'CMPS',
                   'CMPX', 'CMRA', 'CMRE', 'CMRX', 'CMS', 'CMT', 'CMTG', 'CMTL', 'CNA', 'CNC', 'CNDA', 'CNDB', 'CNDT',
                   'CNET', 'CNEY', 'CNF', 'CNFR', 'CNGL', 'CNHI', 'CNI', 'CNK', 'CNM', 'CNMD', 'CNNE', 'CNO', 'CNOB',
                   'CNP', 'CNQ', 'CNS', 'CNSL', 'CNSP', 'CNTA', 'CNTB', 'CNTG', 'CNTX', 'CNTY', 'CNX', 'CNXA', 'CNXC',
                   'CNXN', 'CO', 'COCO', 'COCP', 'CODA', 'CODI', 'CODX', 'COE', 'COEP', 'COF', 'COFS', 'COGT', 'COHN',
                   'COHR', 'COHU', 'COIN', 'COKE', 'COLB', 'COLD', 'COLL', 'COLM', 'COMM', 'COMP', 'COMS', 'CONN',
                   'CONX', 'COO', 'COOK', 'COOL', 'COOP', 'COP', 'CORR', 'CORS', 'CORT', 'COSM', 'COST', 'COTY', 'COUR',
                   'COYA', 'CP', 'CPA', 'CPAA', 'CPAC', 'CPB', 'CPE', 'CPF', 'CPG', 'CPHC', 'CPHI', 'CPIX', 'CPK',
                   'CPLP', 'CPNG', 'CPOP', 'CPRI', 'CPRT', 'CPRX', 'CPS', 'CPSH', 'CPSI', 'CPSS', 'CPT', 'CPTK', 'CPTN',
                   'CPUH', 'CQP', 'CR', 'CRAI', 'CRBG', 'CRBP', 'CRBU', 'CRC', 'CRCT', 'CRD.A', 'CRD.B', 'CRDF', 'CRDL',
                   'CRDO', 'CREC', 'CREG', 'CRESY', 'CREX', 'CRGE', 'CRGO', 'CRGY', 'CRH', 'CRI', 'CRIS', 'CRK', 'CRKN',
                   'CRL', 'CRM', 'CRMD', 'CRMT', 'CRNC', 'CRNT', 'CRNX', 'CRON', 'CROX', 'CRS', 'CRSP', 'CRSR', 'CRT',
                   'CRTO', 'CRUS', 'CRVL', 'CRVS', 'CRWD', 'CRWS', 'CRZN', 'CS', 'CSAN', 'CSBR', 'CSCO', 'CSGP', 'CSGS',
                   'CSII', 'CSIQ', 'CSL', 'CSLM', 'CSPI', 'CSR', 'CSSE', 'CSTA', 'CSTE', 'CSTL', 'CSTM', 'CSTR', 'CSV',
                   'CSWC', 'CSWI', 'CSX', 'CTAS', 'CTBI', 'CTG', 'CTGO', 'CTHR', 'CTIB', 'CTIC', 'CTKB', 'CTLP', 'CTLT',
                   'CTM', 'CTMX', 'CTO', 'CTOS', 'CTRA', 'CTRE', 'CTRM', 'CTRN', 'CTS', 'CTSH', 'CTSO', 'CTV', 'CTVA',
                   'CTXR', 'CUBE', 'CUBI', 'CUE', 'CUEN', 'CUK', 'CULL', 'CULP', 'CURI', 'CURO', 'CURV', 'CUTR', 'CUZ',
                   'CVAC', 'CVBF', 'CVCO', 'CVCY', 'CVE', 'CVEO', 'CVGI', 'CVGW', 'CVI', 'CVII', 'CVKD', 'CVLG', 'CVLT',
                   'CVLY', 'CVM', 'CVNA', 'CVR', 'CVRX', 'CVS', 'CVT', 'CVU', 'CVV', 'CVX', 'CW', 'CWAN', 'CWBC',
                   'CWBR', 'CWCO', 'CWEN', 'CWEN.A', 'CWH', 'CWK', 'CWST', 'CWT', 'CX', 'CXAC', 'CXAI', 'CXDO', 'CXM',
                   'CXW', 'CYAD', 'CYAN', 'CYBN', 'CYBR', 'CYCC', 'CYCN', 'CYD', 'CYH', 'CYN', 'CYRX', 'CYT', 'CYTH',
                   'CYTK', 'CYTO', 'CYXT', 'CZFS', 'CZNC', 'CZOO', 'CZR', 'CZWI', 'D', 'DAC', 'DADA', 'DAIO', 'DAKT',
                   'DAL', 'DALN', 'DALS', 'DAN', 'DAO', 'DAR', 'DARE', 'DASH', 'DATS', 'DAVA', 'DAVE', 'DAWN', 'DB',
                   'DBD', 'DBGI', 'DBI', 'DBRG', 'DBTX', 'DBVT', 'DBX', 'DC', 'DCBO', 'DCFC', 'DCGO', 'DCI', 'DCO',
                   'DCOM', 'DCP', 'DCPH', 'DCT', 'DCTH', 'DD', 'DDD', 'DDI', 'DDL', 'DDOG', 'DDS', 'DE', 'DEA', 'DECA',
                   'DECK', 'DEI', 'DELL', 'DEN', 'DENN', 'DEO', 'DERM', 'DESP', 'DFFN', 'DFH', 'DFIN', 'DFLI', 'DFS',
                   'DG', 'DGHI', 'DGICA', 'DGICB', 'DGII', 'DGLY', 'DGX', 'DH', 'DHAC', 'DHC', 'DHCA', 'DHHC', 'DHI',
                   'DHIL', 'DHR', 'DHT', 'DHX', 'DIBS', 'DICE', 'DIN', 'DINO', 'DIOD', 'DIS', 'DISA', 'DISH', 'DIST',
                   'DIT', 'DJCO', 'DK', 'DKDCA', 'DKL', 'DKNG', 'DKS', 'DLA', 'DLB', 'DLHC', 'DLNG', 'DLO', 'DLPN',
                   'DLR', 'DLTH', 'DLTR', 'DLX', 'DM', 'DMAC', 'DMAQ', 'DMLP', 'DMRC', 'DMS', 'DMTK', 'DMYS', 'DMYY',
                   'DNA', 'DNAB', 'DNAD', 'DNB', 'DNLI', 'DNMR', 'DNN', 'DNOW', 'DNUT', 'DO', 'DOC', 'DOCN', 'DOCS',
                   'DOCU', 'DOGZ', 'DOLE', 'DOMA', 'DOMH', 'DOMO', 'DOOO', 'DOOR', 'DORM', 'DOUG', 'DOV', 'DOW', 'DOX',
                   'DOYU', 'DPCS', 'DPRO', 'DPSI', 'DPZ', 'DQ', 'DRCT', 'DRD', 'DRH', 'DRI', 'DRIO', 'DRMA', 'DRQ',
                   'DRRX', 'DRS', 'DRTS', 'DRTT', 'DRUG', 'DRVN', 'DSAQ', 'DSEY', 'DSGN', 'DSGR', 'DSGX', 'DSKE', 'DSP',
                   'DSS', 'DSWL', 'DSX', 'DT', 'DTC', 'DTE', 'DTEA', 'DTIL', 'DTM', 'DTOC', 'DTSS', 'DTST', 'DUET',
                   'DUK', 'DUNE', 'DUO', 'DUOL', 'DUOT', 'DV', 'DVA', 'DVAX', 'DVN', 'DWAC', 'DWSN', 'DX', 'DXC',
                   'DXCM', 'DXF', 'DXLG', 'DXPE', 'DXR', 'DXYN', 'DY', 'DYAI', 'DYN', 'DYNT', 'DZSI', 'E', 'EA', 'EAC',
                   'EAF', 'EAR', 'EARN', 'EAST', 'EAT', 'EB', 'EBAY', 'EBC', 'EBET', 'EBF', 'EBIX', 'EBMT', 'EBON',
                   'EBR', 'EBS', 'EBTC', 'EC', 'ECBK', 'ECC', 'ECL', 'ECOR', 'ECPG', 'ECVT', 'ECX', 'ED', 'EDAP',
                   'EDBL', 'EDIT', 'EDN', 'EDR', 'EDRY', 'EDSA', 'EDTK', 'EDTX', 'EDU', 'EDUC', 'EE', 'EEFT', 'EEIQ',
                   'EEX', 'EFC', 'EFHT', 'EFOI', 'EFSC', 'EFSH', 'EFTR', 'EFX', 'EFXT', 'EGAN', 'EGBN', 'EGGF', 'EGHT',
                   'EGIO', 'EGLE', 'EGLX', 'EGO', 'EGP', 'EGRX', 'EGY', 'EH', 'EHAB', 'EHC', 'EHTH', 'EIC', 'EIG',
                   'EIGR', 'EIX', 'EJH', 'EKSO', 'EL', 'ELA', 'ELAN', 'ELBM', 'ELDN', 'ELEV', 'ELF', 'ELLO', 'ELMD',
                   'ELME', 'ELOX', 'ELP', 'ELS', 'ELSE', 'ELTK', 'ELV', 'ELVN', 'ELYM', 'ELYS', 'EM', 'EMAN', 'EMBC',
                   'EMBK', 'EMCG', 'EME', 'EMKR', 'EML', 'EMLD', 'EMN', 'EMR', 'EMX', 'ENB', 'ENCP', 'ENER', 'ENFN',
                   'ENG', 'ENIC', 'ENLC', 'ENLT', 'ENLV', 'ENOB', 'ENOV', 'ENPH', 'ENR', 'ENS', 'ENSC', 'ENSG', 'ENSV',
                   'ENTA', 'ENTF', 'ENTG', 'ENTX', 'ENV', 'ENVA', 'ENVB', 'ENVX', 'ENZ', 'EOCW', 'EOG', 'EOLS', 'EOSE',
                   'EP', 'EPAC', 'EPAM', 'EPC', 'EPD', 'EPIX', 'EPM', 'EPOW', 'EPR', 'EPRT', 'EPSN', 'EQ', 'EQBK',
                   'EQC', 'EQH', 'EQIX', 'EQNR', 'EQR', 'EQRX', 'EQS', 'EQT', 'EQX', 'ERAS', 'ERES', 'ERF', 'ERIC',
                   'ERIE', 'ERII', 'ERJ', 'ERNA', 'ERO', 'ERYP', 'ES', 'ESAB', 'ESAC', 'ESBA', 'ESCA', 'ESE', 'ESEA',
                   'ESGR', 'ESI', 'ESLT', 'ESMT', 'ESNT', 'ESOA', 'ESP', 'ESPR', 'ESQ', 'ESRT', 'ESS', 'ESSA', 'ESTA',
                   'ESTC', 'ESTE', 'ET', 'ETAO', 'ETD', 'ETN', 'ETNB', 'ETON', 'ETR', 'ETRN', 'ETSY', 'ETWO', 'EU',
                   'EUCR', 'EUDA', 'EURN', 'EVA', 'EVAX', 'EVBG', 'EVBN', 'EVC', 'EVCM', 'EVE', 'EVER', 'EVEX', 'EVGN',
                   'EVGO', 'EVGR', 'EVH', 'EVI', 'EVLO', 'EVLV', 'EVO', 'EVOJ', 'EVOK', 'EVOP', 'EVR', 'EVRG', 'EVRI',
                   'EVTC', 'EVTL', 'EVTV', 'EW', 'EWBC', 'EWCZ', 'EWTX', 'EXAI', 'EXAS', 'EXC', 'EXEL', 'EXFY', 'EXK',
                   'EXLS', 'EXP', 'EXPD', 'EXPE', 'EXPI', 'EXPO', 'EXPR', 'EXR', 'EXTR', 'EYE', 'EYEN', 'EYPT', 'EZFL',
                   'EZGO', 'EZPW', 'F', 'FA', 'FACT', 'FAF', 'FAMI', 'FANG', 'FANH', 'FARM', 'FARO', 'FAST', 'FAT',
                   'FATBB', 'FATE', 'FATH', 'FATP', 'FAZE', 'FBIN', 'FBIO', 'FBIZ', 'FBK', 'FBMS', 'FBNC', 'FBP',
                   'FBRT', 'FBRX', 'FC', 'FCAP', 'FCBC', 'FCCO', 'FCEL', 'FCF', 'FCFS', 'FCN', 'FCNCA', 'FCPT', 'FCUV',
                   'FCX', 'FDBC', 'FDMT', 'FDP', 'FDS', 'FDUS', 'FDX', 'FE', 'FEAM', 'FEDU', 'FEIM', 'FELE', 'FEMY',
                   'FENC', 'FENG', 'FERG', 'FET', 'FEXD', 'FF', 'FFBC', 'FFIC', 'FFIE', 'FFIN', 'FFIV', 'FFNW', 'FFWM',
                   'FG', 'FGBI', 'FGEN', 'FGF', 'FGH', 'FGI', 'FGMC', 'FHB', 'FHI', 'FHLT', 'FHN', 'FHTX', 'FIAC',
                   'FIBK', 'FICO', 'FICV', 'FIGS', 'FINV', 'FINW', 'FIP', 'FIS', 'FISI', 'FISV', 'FITB', 'FIVE', 'FIVN',
                   'FIX', 'FIXX', 'FIZZ', 'FKWL', 'FL', 'FLAG', 'FLEX', 'FLFV', 'FLGC', 'FLGT', 'FLIC', 'FLJ', 'FLL',
                   'FLME', 'FLNC', 'FLNG', 'FLNT', 'FLO', 'FLR', 'FLS', 'FLT', 'FLUX', 'FLWS', 'FLXS', 'FLYW', 'FMAO',
                   'FMBH', 'FMC', 'FMIV', 'FMNB', 'FMS', 'FMX', 'FN', 'FNA', 'FNB', 'FNCB', 'FNCH', 'FND', 'FNF',
                   'FNGR', 'FNKO', 'FNLC', 'FNV', 'FNVT', 'FNWB', 'FNWD', 'FOA', 'FOCS', 'FOLD', 'FONR', 'FOR', 'FORA',
                   'FORD', 'FORG', 'FORL', 'FORM', 'FORR', 'FORTY', 'FOSL', 'FOUR', 'FOX', 'FOXA', 'FOXF', 'FOXO',
                   'FPAY', 'FPH', 'FPI', 'FR', 'FRAF', 'FRBA', 'FRBK', 'FRBN', 'FRC', 'FRD', 'FREE', 'FREQ', 'FREY',
                   'FRG', 'FRGE', 'FRGI', 'FRGT', 'FRHC', 'FRLA', 'FRLN', 'FRME', 'FRO', 'FROG', 'FRPH', 'FRPT', 'FRSH',
                   'FRST', 'FRSX', 'FRT', 'FRTX', 'FRXB', 'FRZA', 'FSBC', 'FSBW', 'FSEA', 'FSFG', 'FSI', 'FSK', 'FSLR',
                   'FSLY', 'FSM', 'FSNB', 'FSP', 'FSR', 'FSRX', 'FSS', 'FSTR', 'FSV', 'FTAI', 'FTCH', 'FTCI', 'FTDR',
                   'FTEK', 'FTFT', 'FTHM', 'FTI', 'FTII', 'FTK', 'FTNT', 'FTS', 'FTV', 'FUBO', 'FUL', 'FULC', 'FULT',
                   'FUN', 'FUNC', 'FURY', 'FUSB', 'FUSN', 'FUTU', 'FUV', 'FVCB', 'FVRR', 'FWAC', 'FWBI', 'FWONA',
                   'FWRD', 'FWRG', 'FXCO', 'FXLV', 'FXNC', 'FYBR', 'FZT', 'G', 'GABC', 'GAIA', 'GAIN', 'GALT', 'GAMB',
                   'GAMC', 'GAME', 'GAN', 'GANX', 'GAQ', 'GASS', 'GATE', 'GATO', 'GATX', 'GAU', 'GB', 'GBBK', 'GBCI',
                   'GBDC', 'GBIO', 'GBLI', 'GBNH', 'GBNY', 'GBR', 'GBRG', 'GBTG', 'GBX', 'GCBC', 'GCI', 'GCMG', 'GCO',
                   'GCT', 'GCTK', 'GD', 'GDC', 'GDDY', 'GDEN', 'GDNR', 'GDOT', 'GDRX', 'GDS', 'GDST', 'GDYN', 'GE',
                   'GECC', 'GEEX', 'GEF', 'GEF.B', 'GEG', 'GEHC', 'GEHI', 'GEL', 'GEN', 'GENC', 'GENE', 'GENI', 'GENQ',
                   'GEO', 'GEOS', 'GERN', 'GES', 'GETR', 'GETY', 'GEVO', 'GFAI', 'GFF', 'GFGD', 'GFI', 'GFL', 'GFOR',
                   'GFS', 'GFX', 'GGAA', 'GGAL', 'GGB', 'GGE', 'GGG', 'GGR', 'GH', 'GHC', 'GHG', 'GHI', 'GHIX', 'GHL',
                   'GHLD', 'GHM', 'GHRS', 'GHSI', 'GIA', 'GIB', 'GIC', 'GIFI', 'GIGM', 'GIII', 'GIL', 'GILD', 'GILT',
                   'GIPR', 'GIS', 'GKOS', 'GL', 'GLAD', 'GLBE', 'GLBS', 'GLBZ', 'GLDD', 'GLDG', 'GLG', 'GLLI', 'GLMD',
                   'GLNG', 'GLOB', 'GLOP', 'GLP', 'GLPG', 'GLPI', 'GLRE', 'GLS', 'GLSI', 'GLST', 'GLT', 'GLTA', 'GLTO',
                   'GLUE', 'GLW', 'GLYC', 'GM', 'GMAB', 'GMBL', 'GMDA', 'GME', 'GMED', 'GMFI', 'GMGI', 'GMRE', 'GMS',
                   'GMVD', 'GNE', 'GNFT', 'GNK', 'GNL', 'GNLN', 'GNLX', 'GNPX', 'GNRC', 'GNS', 'GNSS', 'GNTA', 'GNTX',
                   'GNTY', 'GNUS', 'GNW', 'GO', 'GOCO', 'GOEV', 'GOGL', 'GOGN', 'GOGO', 'GOL', 'GOLD', 'GOLF', 'GOOD',
                   'GOOG', 'GOOGL', 'GOOS', 'GORO', 'GOSS', 'GOTU', 'GOVX', 'GP', 'GPAC', 'GPC', 'GPCR', 'GPI', 'GPK',
                   'GPMT', 'GPN', 'GPOR', 'GPP', 'GPRE', 'GPRK', 'GPRO', 'GPS', 'GRAB', 'GRBK', 'GRC', 'GRCL', 'GRCY',
                   'GREE', 'GRFS', 'GRFX', 'GRIL', 'GRIN', 'GRMN', 'GRNA', 'GRND', 'GRNQ', 'GRNT', 'GROM', 'GROV',
                   'GROW', 'GROY', 'GRPH', 'GRPN', 'GRRR', 'GRTS', 'GRTX', 'GRVY', 'GRWG', 'GS', 'GSAT', 'GSBC', 'GSBD',
                   'GSD', 'GSHD', 'GSIT', 'GSK', 'GSL', 'GSM', 'GSMG', 'GSQB', 'GSRM', 'GSUN', 'GT', 'GTAC', 'GTBP',
                   'GTE', 'GTEC', 'GTES', 'GTH', 'GTHX', 'GTIM', 'GTLB', 'GTLS', 'GTN', 'GTN.A', 'GTX', 'GTY', 'GURE',
                   'GVA', 'GVCI', 'GVP', 'GWAV', 'GWH', 'GWRE', 'GWRS', 'GWW', 'GXO', 'GYRO', 'H', 'HA', 'HAE', 'HAFC',
                   'HAIA', 'HAIN', 'HAL', 'HALL', 'HALO', 'HARP', 'HAS', 'HASI', 'HAYN', 'HAYW', 'HBAN', 'HBB', 'HBCP',
                   'HBI', 'HBIO', 'HBM', 'HBNC', 'HBT', 'HCA', 'HCAT', 'HCC', 'HCCI', 'HCDI', 'HCI', 'HCKT', 'HCM',
                   'HCMA', 'HCNE', 'HCP', 'HCSG', 'HCTI', 'HCVI', 'HCWB', 'HD', 'HDB', 'HDSN', 'HE', 'HEAR', 'HEES',
                   'HEI', 'HEI.A', 'HELE', 'HEP', 'HEPA', 'HEPS', 'HES', 'HESM', 'HEXO', 'HFBL', 'HFFG', 'HFWA', 'HGBL',
                   'HGEN', 'HGTY', 'HGV', 'HHC', 'HHGC', 'HHLA', 'HHRS', 'HHS', 'HI', 'HIBB', 'HIFS', 'HIG', 'HIHO',
                   'HII', 'HILS', 'HIMS', 'HIMX', 'HIPO', 'HITI', 'HIVE', 'HIW', 'HKD', 'HL', 'HLBZ', 'HLF', 'HLGN',
                   'HLI', 'HLIO', 'HLIT', 'HLLY', 'HLMN', 'HLN', 'HLNE', 'HLT', 'HLTH', 'HLVX', 'HLX', 'HMA', 'HMAC',
                   'HMC', 'HMN', 'HMNF', 'HMPT', 'HMST', 'HMY', 'HNI', 'HNNA', 'HNRA', 'HNRG', 'HNST', 'HNVR', 'HOFT',
                   'HOFV', 'HOG', 'HOLI', 'HOLO', 'HOLX', 'HOMB', 'HON', 'HONE', 'HOOD', 'HOOK', 'HOPE', 'HOTH', 'HOUR',
                   'HOUS', 'HOV', 'HOWL', 'HP', 'HPCO', 'HPE', 'HPK', 'HPLT', 'HPP', 'HPQ', 'HQI', 'HQY', 'HR', 'HRB',
                   'HRI', 'HRL', 'HRMY', 'HROW', 'HRT', 'HRTG', 'HRTX', 'HRZN', 'HSAI', 'HSBC', 'HSC', 'HSCS', 'HSDT',
                   'HSIC', 'HSII', 'HSKA', 'HSON', 'HSPO', 'HST', 'HSTM', 'HSTO', 'HSY', 'HT', 'HTBI', 'HTBK', 'HTCR',
                   'HTGC', 'HTGM', 'HTH', 'HTHT', 'HTLD', 'HTLF', 'HTOO', 'HTZ', 'HUBB', 'HUBC', 'HUBG', 'HUBS', 'HUDA',
                   'HUDI', 'HUGE', 'HUIZ', 'HUM', 'HUMA', 'HUN', 'HURC', 'HURN', 'HUSA', 'HUT', 'HUYA', 'HVBC', 'HVT',
                   'HWBK', 'HWC', 'HWEL', 'HWKN', 'HWKZ', 'HWM', 'HXL', 'HY', 'HYFM', 'HYLN', 'HYMC', 'HYPR', 'HYW',
                   'HYZN', 'HZNP', 'HZO', 'HZON', 'IAC', 'IAG', 'IART', 'IAS', 'IAUX', 'IBA', 'IBCP', 'IBEX', 'IBIO',
                   'IBKR', 'IBM', 'IBN', 'IBOC', 'IBP', 'IBRX', 'IBTX', 'ICAD', 'ICCC', 'ICCH', 'ICCM', 'ICD', 'ICE',
                   'ICFI', 'ICG', 'ICHR', 'ICL', 'ICLK', 'ICLR', 'ICMB', 'ICNC', 'ICPT', 'ICU', 'ICUI', 'ICVX', 'ID',
                   'IDA', 'IDAI', 'IDBA', 'IDCC', 'IDEX', 'IDN', 'IDR', 'IDT', 'IDW', 'IDXX', 'IDYA', 'IE', 'IEP',
                   'IESC', 'IEX', 'IFBD', 'IFF', 'IFIN', 'IFRX', 'IFS', 'IGC', 'IGIC', 'IGMS', 'IGT', 'IGTA', 'IH',
                   'IHG', 'IHRT', 'IHS', 'IHT', 'III', 'IIIN', 'IIIV', 'IINN', 'IIPR', 'IKNA', 'IKT', 'ILAG', 'ILMN',
                   'ILPT', 'IMAB', 'IMAQ', 'IMAX', 'IMBI', 'IMCC', 'IMCR', 'IMGN', 'IMH', 'IMKTA', 'IMMP', 'IMMR',
                   'IMMX', 'IMNM', 'IMNN', 'IMO', 'IMOS', 'IMPL', 'IMPP', 'IMRN', 'IMRX', 'IMTE', 'IMTX', 'IMUX', 'IMV',
                   'IMVT', 'IMXI', 'INAB', 'INAQ', 'INBK', 'INBS', 'INBX', 'INCR', 'INCY', 'INDB', 'INDI', 'INDO',
                   'INDP', 'INDT', 'INFA', 'INFI', 'INFN', 'INFU', 'INFY', 'ING', 'INGN', 'INGR', 'INKT', 'INLX', 'INM',
                   'INMB', 'INMD', 'INN', 'INNV', 'INO', 'INOD', 'INPX', 'INSE', 'INSG', 'INSM', 'INSP', 'INST', 'INSW',
                   'INT', 'INTA', 'INTC', 'INTE', 'INTG', 'INTR', 'INTT', 'INTU', 'INTZ', 'INUV', 'INVA', 'INVE',
                   'INVH', 'INVO', 'INVZ', 'INZY', 'IOAC', 'IOBT', 'IONM', 'IONQ', 'IONR', 'IONS', 'IOR', 'IOSP', 'IOT',
                   'IOVA', 'IP', 'IPA', 'IPAR', 'IPDN', 'IPG', 'IPGP', 'IPHA', 'IPI', 'IPSC', 'IPVF', 'IPVI', 'IPW',
                   'IPWR', 'IPX', 'IQ', 'IQMD', 'IQV', 'IR', 'IRAA', 'IRBT', 'IRDM', 'IREN', 'IRIX', 'IRM', 'IRMD',
                   'IRNT', 'IRON', 'IROQ', 'IRRX', 'IRS', 'IRT', 'IRTC', 'IRWD', 'ISDR', 'ISEE', 'ISIG', 'ISPC', 'ISPO',
                   'ISRG', 'ISRL', 'ISSC', 'ISTR', 'ISUN', 'IT', 'ITAQ', 'ITCB', 'ITCI', 'ITGR', 'ITI', 'ITIC', 'ITOS',
                   'ITP', 'ITRG', 'ITRI', 'ITRM', 'ITRN', 'ITT', 'ITUB', 'ITW', 'IVA', 'IVAC', 'IVCA', 'IVCB', 'IVCP',
                   'IVDA', 'IVR', 'IVT', 'IVVD', 'IVZ', 'IX', 'IXAQ', 'IXHL', 'IZEA', 'IZM', 'J', 'JACK', 'JAGX',
                   'JAKK', 'JAMF', 'JAN', 'JANX', 'JAQC', 'JAZZ', 'JBGS', 'JBHT', 'JBI', 'JBL', 'JBLU', 'JBSS', 'JBT',
                   'JCI', 'JCSE', 'JCTCF', 'JD', 'JEF', 'JELD', 'JEWL', 'JFBR', 'JFIN', 'JFU', 'JG', 'JGGC', 'JHG',
                   'JHX', 'JILL', 'JJSF', 'JKHY', 'JKS', 'JLL', 'JMAC', 'JMIA', 'JMSB', 'JNCE', 'JNJ', 'JNPR', 'JOAN',
                   'JOB', 'JOBY', 'JOE', 'JOUT', 'JPM', 'JRSH', 'JRVR', 'JSPR', 'JT', 'JUGG', 'JUN', 'JUPW', 'JVA',
                   'JWAC', 'JWEL', 'JWN', 'JWSM', 'JXJT', 'JXN', 'JYNT', 'JZ', 'JZXN', 'K', 'KA', 'KACL', 'KAI', 'KAL',
                   'KALA', 'KALU', 'KALV', 'KAMN', 'KAR', 'KARO', 'KAVL', 'KB', 'KBAL', 'KBH', 'KBNT', 'KBR', 'KC',
                   'KCGI', 'KD', 'KDNY', 'KDP', 'KE', 'KELYA', 'KELYB', 'KEN', 'KEP', 'KEQU', 'KERN', 'KEX', 'KEY',
                   'KEYS', 'KFFB', 'KFRC', 'KFS', 'KFY', 'KGC', 'KHC', 'KIDS', 'KIM', 'KIND', 'KINS', 'KIQ', 'KIRK',
                   'KITT', 'KKR', 'KLAC', 'KLIC', 'KLR', 'KLTR', 'KLXE', 'KMB', 'KMDA', 'KMI', 'KMPR', 'KMT', 'KMX',
                   'KN', 'KNDI', 'KNOP', 'KNSA', 'KNSL', 'KNSW', 'KNTE', 'KNTK', 'KNW', 'KNX', 'KO', 'KOD', 'KODK',
                   'KOF', 'KOP', 'KOPN', 'KORE', 'KOS', 'KOSS', 'KPLT', 'KPRX', 'KPTI', 'KR', 'KRBP', 'KRC', 'KREF',
                   'KRG', 'KRKR', 'KRMD', 'KRNL', 'KRNT', 'KRNY', 'KRO', 'KRON', 'KROS', 'KRP', 'KRT', 'KRTX', 'KRUS',
                   'KRYS', 'KSCP', 'KSPN', 'KSS', 'KT', 'KTB', 'KTCC', 'KTOS', 'KTRA', 'KTTA', 'KUKE', 'KULR', 'KURA',
                   'KVHI', 'KVSA', 'KVSC', 'KW', 'KWE', 'KWR', 'KXIN', 'KYCH', 'KYMR', 'KZIA', 'KZR', 'L', 'LAB',
                   'LABP', 'LAC', 'LAD', 'LADR', 'LAKE', 'LAMR', 'LANC', 'LAND', 'LANV', 'LARK', 'LASE', 'LASR', 'LATG',
                   'LAUR', 'LAW', 'LAZ', 'LAZR', 'LAZY', 'LBAI', 'LBBB', 'LBC', 'LBPH', 'LBRDA', 'LBRDK', 'LBRT',
                   'LBTYA', 'LBTYB', 'LBTYK', 'LC', 'LCA', 'LCAA', 'LCFY', 'LCI', 'LCID', 'LCII', 'LCNB', 'LCTX',
                   'LCUT', 'LCW', 'LDI', 'LDOS', 'LE', 'LEA', 'LECO', 'LEDS', 'LEE', 'LEG', 'LEGH', 'LEGN', 'LEJU',
                   'LEN', 'LEN.B', 'LESL', 'LEU', 'LEV', 'LEVI', 'LEXX', 'LFAC', 'LFCR', 'LFLY', 'LFMD', 'LFST', 'LFT',
                   'LFUS', 'LFVN', 'LGF.A', 'LGF.B', 'LGHL', 'LGIH', 'LGL', 'LGMK', 'LGND', 'LGO', 'LGST', 'LGVC',
                   'LGVN', 'LH', 'LHC', 'LHX', 'LI', 'LIAN', 'LIBY', 'LICN', 'LICY', 'LIDR', 'LIFE', 'LIFW', 'LII',
                   'LILA', 'LILAK', 'LILM', 'LIN', 'LINC', 'LIND', 'LINK', 'LIPO', 'LIQT', 'LITB', 'LITE', 'LITM',
                   'LITT', 'LIVB', 'LIVE', 'LIVN', 'LIXT', 'LIZI', 'LKCO', 'LKFN', 'LKQ', 'LL', 'LLAP', 'LLY', 'LMAT',
                   'LMB', 'LMDX', 'LMFA', 'LMND', 'LMNL', 'LMNR', 'LMST', 'LMT', 'LNC', 'LND', 'LNG', 'LNKB', 'LNN',
                   'LNSR', 'LNT', 'LNTH', 'LNW', 'LNZA', 'LOAN', 'LOB', 'LOCC', 'LOCL', 'LOCO', 'LODE', 'LOGI', 'LOMA',
                   'LOOP', 'LOPE', 'LOV', 'LOVE', 'LOW', 'LPCN', 'LPG', 'LPL', 'LPLA', 'LPRO', 'LPSN', 'LPTH', 'LPTV',
                   'LPTX', 'LPX', 'LQDA', 'LQDT', 'LRCX', 'LRFC', 'LRMR', 'LRN', 'LSAK', 'LSBK', 'LSCC', 'LSDI', 'LSEA',
                   'LSF', 'LSI', 'LSPD', 'LSTA', 'LSTR', 'LTBR', 'LTC', 'LTCH', 'LTH', 'LTHM', 'LTRN', 'LTRPA', 'LTRPB',
                   'LTRX', 'LTRY', 'LU', 'LUCD', 'LUCY', 'LULU', 'LUMN', 'LUMO', 'LUNA', 'LUNG', 'LUNR', 'LUV', 'LUXH',
                   'LVAC', 'LVLU', 'LVO', 'LVOX', 'LVRO', 'LVS', 'LVTX', 'LVWR', 'LW', 'LWAY', 'LWLG', 'LX', 'LXEH',
                   'LXFR', 'LXP', 'LXRX', 'LXU', 'LYB', 'LYEL', 'LYFT', 'LYG', 'LYRA', 'LYT', 'LYTS', 'LYV', 'LZ',
                   'LZB', 'M', 'MA', 'MAA', 'MAC', 'MACA', 'MACK', 'MAG', 'MAIA', 'MAIN', 'MAN', 'MANH', 'MANU', 'MAPS',
                   'MAQC', 'MAR', 'MARA', 'MARK', 'MARPS', 'MARX', 'MAS', 'MASI', 'MASS', 'MAT', 'MATH', 'MATV', 'MATW',
                   'MATX', 'MAX', 'MAXN', 'MAXR', 'MAYS', 'MBAC', 'MBC', 'MBCN', 'MBI', 'MBIN', 'MBIO', 'MBLY', 'MBOT',
                   'MBRX', 'MBSC', 'MBTC', 'MBUU', 'MBWM', 'MC', 'MCAA', 'MCAC', 'MCAF', 'MCAG', 'MCB', 'MCBC', 'MCBS',
                   'MCD', 'MCFT', 'MCHP', 'MCHX', 'MCK', 'MCLD', 'MCO', 'MCRB', 'MCRI', 'MCS', 'MCVT', 'MCW', 'MCY',
                   'MD', 'MDB', 'MDC', 'MDGL', 'MDGS', 'MDIA', 'MDJH', 'MDLZ', 'MDNA', 'MDRR', 'MDRX', 'MDT', 'MDU',
                   'MDV', 'MDVL', 'MDWD', 'MDWT', 'MDXG', 'MDXH', 'ME', 'MEC', 'MED', 'MEDP', 'MEDS', 'MEG', 'MEGL',
                   'MEI', 'MEIP', 'MEKA', 'MELI', 'MEOA', 'MEOH', 'MERC', 'MESA', 'MESO', 'MET', 'META', 'METC', 'METX',
                   'MF', 'MFA', 'MFC', 'MFG', 'MFIC', 'MFIN', 'MG', 'MGA', 'MGAM', 'MGEE', 'MGI', 'MGIC', 'MGLD', 'MGM',
                   'MGNI', 'MGNX', 'MGOL', 'MGPI', 'MGRC', 'MGRX', 'MGTA', 'MGTX', 'MGY', 'MGYR', 'MHH', 'MHK', 'MHLD',
                   'MHO', 'MHUA', 'MICS', 'MIDD', 'MIGI', 'MIMO', 'MIND', 'MINM', 'MIR', 'MIRM', 'MIRO', 'MIST', 'MITA',
                   'MITK', 'MITQ', 'MITT', 'MIXT', 'MKC', 'MKFG', 'MKL', 'MKSI', 'MKTW', 'MKTX', 'MKUL', 'ML', 'MLAB',
                   'MLAC', 'MLCO', 'MLEC', 'MLGO', 'MLI', 'MLKN', 'MLM', 'MLNK', 'MLP', 'MLR', 'MLSS', 'MLTX', 'MLVF',
                   'MLYS', 'MMAT', 'MMC', 'MMI', 'MMLP', 'MMM', 'MMMB', 'MMP', 'MMS', 'MMSI', 'MMV', 'MMYT', 'MNDO',
                   'MNDY', 'MNKD', 'MNMD', 'MNOV', 'MNPR', 'MNRO', 'MNSB', 'MNSO', 'MNST', 'MNTK', 'MNTN', 'MNTS',
                   'MNTV', 'MNTX', 'MO', 'MOB', 'MOBQ', 'MOBV', 'MOD', 'MODD', 'MODG', 'MODN', 'MODV', 'MOFG', 'MOG.A',
                   'MOG.B', 'MOGO', 'MOGU', 'MOH', 'MOLN', 'MOMO', 'MOND', 'MOR', 'MORF', 'MORN', 'MOS', 'MOTS', 'MOV',
                   'MOVE', 'MOXC', 'MP', 'MPAA', 'MPB', 'MPC', 'MPLN', 'MPLX', 'MPRA', 'MPTI', 'MPU', 'MPW', 'MPWR',
                   'MPX', 'MQ', 'MRAI', 'MRAM', 'MRBK', 'MRC', 'MRCC', 'MRCY', 'MRDB', 'MREO', 'MRIN', 'MRK', 'MRKR',
                   'MRM', 'MRNA', 'MRNS', 'MRO', 'MRSN', 'MRTN', 'MRTX', 'MRUS', 'MRVI', 'MRVL', 'MS', 'MSA', 'MSAC',
                   'MSB', 'MSBI', 'MSC', 'MSCI', 'MSDA', 'MSEX', 'MSFT', 'MSGE', 'MSGM', 'MSGS', 'MSI', 'MSM', 'MSN',
                   'MSSA', 'MSTR', 'MSVB', 'MT', 'MTA', 'MTAC', 'MTAL', 'MTB', 'MTC', 'MTCH', 'MTCR', 'MTD', 'MTDR',
                   'MTEK', 'MTEM', 'MTEX', 'MTG', 'MTH', 'MTLS', 'MTN', 'MTNB', 'MTP', 'MTR', 'MTRN', 'MTRX', 'MTRY',
                   'MTSI', 'MTTR', 'MTVC', 'MTW', 'MTX', 'MTZ', 'MU', 'MUFG', 'MULN', 'MUR', 'MURF', 'MUSA', 'MUX',
                   'MVBF', 'MVIS', 'MVLA', 'MVST', 'MWA', 'MX', 'MXC', 'MXCT', 'MXL', 'MYE', 'MYFW', 'MYGN', 'MYMD',
                   'MYNA', 'MYNZ', 'MYO', 'MYPS', 'MYRG', 'MYSZ', 'MYTE', 'NA', 'NAAS', 'NABL', 'NAII', 'NAK', 'NAMS',
                   'NAOV', 'NAPA', 'NARI', 'NAT', 'NATH', 'NATI', 'NATR', 'NAUT', 'NAVB', 'NAVI', 'NB', 'NBHC', 'NBIX',
                   'NBN', 'NBR', 'NBRV', 'NBSE', 'NBST', 'NBTB', 'NBTX', 'NBY', 'NC', 'NCAC', 'NCLH', 'NCMI', 'NCNA',
                   'NCNO', 'NCPL', 'NCR', 'NCRA', 'NCSM', 'NCTY', 'NDAQ', 'NDLS', 'NDRA', 'NDSN', 'NE', 'NECB', 'NEE',
                   'NEGG', 'NEM', 'NEN', 'NEO', 'NEOG', 'NEON', 'NEOV', 'NEP', 'NEPH', 'NEPT', 'NERV', 'NESR', 'NET',
                   'NETC', 'NETI', 'NEU', 'NEWP', 'NEWR', 'NEWT', 'NEX', 'NEXA', 'NEXI', 'NEXT', 'NFBK', 'NFE', 'NFG',
                   'NFGC', 'NFLX', 'NFNT', 'NFTG', 'NFYS', 'NG', 'NGC', 'NGD', 'NGG', 'NGL', 'NGM', 'NGMS', 'NGS',
                   'NGVC', 'NGVT', 'NH', 'NHC', 'NHI', 'NHIC', 'NHTC', 'NHWK', 'NI', 'NIC', 'NICE', 'NICK', 'NINE',
                   'NIO', 'NIR', 'NISN', 'NIU', 'NJR', 'NKE', 'NKLA', 'NKSH', 'NKTR', 'NKTX', 'NL', 'NLS', 'NLSP',
                   'NLTX', 'NLY', 'NM', 'NMFC', 'NMG', 'NMIH', 'NMM', 'NMR', 'NMRD', 'NMRK', 'NMTC', 'NMTR', 'NN',
                   'NNBR', 'NNDM', 'NNI', 'NNN', 'NNOX', 'NNVC', 'NOA', 'NOAH', 'NOC', 'NODK', 'NOG', 'NOGN', 'NOK',
                   'NOMD', 'NOTE', 'NOTV', 'NOV', 'NOVA', 'NOVN', 'NOVT', 'NOVV', 'NOW', 'NPAB', 'NPCE', 'NPK', 'NPO',
                   'NR', 'NRAC', 'NRBO', 'NRC', 'NRDS', 'NRDY', 'NREF', 'NRG', 'NRGV', 'NRIM', 'NRIX', 'NRP', 'NRSN',
                   'NRT', 'NRXP', 'NS', 'NSA', 'NSC', 'NSIT', 'NSP', 'NSPR', 'NSSC', 'NSTB', 'NSTC', 'NSTD', 'NSTG',
                   'NSTS', 'NSYS', 'NTAP', 'NTB', 'NTCO', 'NTCT', 'NTES', 'NTGR', 'NTIC', 'NTIP', 'NTLA', 'NTNX', 'NTR',
                   'NTRA', 'NTRB', 'NTRS', 'NTST', 'NTWK', 'NTZ', 'NU', 'NUBI', 'NUE', 'NURO', 'NUS', 'NUTX', 'NUVA',
                   'NUVB', 'NUVL', 'NUWE', 'NUZE', 'NVAC', 'NVAX', 'NVCN', 'NVCR', 'NVCT', 'NVDA', 'NVEC', 'NVEE',
                   'NVEI', 'NVFY', 'NVGS', 'NVIV', 'NVMI', 'NVNO', 'NVO', 'NVOS', 'NVR', 'NVRO', 'NVS', 'NVST', 'NVT',
                   'NVTA', 'NVTS', 'NVVE', 'NVX', 'NWBI', 'NWE', 'NWFL', 'NWG', 'NWL', 'NWLI', 'NWN', 'NWPX', 'NWS',
                   'NWSA', 'NWTN', 'NX', 'NXE', 'NXGL', 'NXGN', 'NXL', 'NXPI', 'NXPL', 'NXRT', 'NXST', 'NXT', 'NXTC',
                   'NXTP', 'NYAX', 'NYC', 'NYCB', 'NYMT', 'NYMX', 'NYT', 'NYXH', 'O', 'OABI', 'OAKUO', 'OB', 'OBE',
                   'OBIO', 'OBLG', 'OBNK', 'OBT', 'OC', 'OCAX', 'OCC', 'OCCI', 'OCEA', 'OCFC', 'OCFT', 'OCG', 'OCGN',
                   'OCN', 'OCS', 'OCSL', 'OCUL', 'OCUP', 'OCX', 'ODC', 'ODFL', 'ODP', 'ODV', 'OEC', 'OESX', 'OFC',
                   'OFED', 'OFG', 'OFIX', 'OFLX', 'OFS', 'OGE', 'OGEN', 'OGI', 'OGN', 'OGS', 'OHAA', 'OHI', 'OI', 'OIG',
                   'OII', 'OIS', 'OKE', 'OKTA', 'OKYO', 'OLB', 'OLED', 'OLIT', 'OLK', 'OLLI', 'OLMA', 'OLN', 'OLO',
                   'OLP', 'OLPX', 'OM', 'OMAB', 'OMC', 'OMCL', 'OMER', 'OMEX', 'OMF', 'OMGA', 'OMH', 'OMI', 'OMIC',
                   'OMQS', 'ON', 'ONB', 'ONCR', 'ONCS', 'ONCT', 'ONCY', 'ONDS', 'ONEW', 'ONFO', 'ONL', 'ONON', 'ONTF',
                   'ONTO', 'ONTX', 'ONVO', 'ONYX', 'OOMA', 'OP', 'OPA', 'OPAD', 'OPAL', 'OPBK', 'OPCH', 'OPEN', 'OPFI',
                   'OPGN', 'OPHC', 'OPI', 'OPK', 'OPOF', 'OPRA', 'OPRT', 'OPRX', 'OPT', 'OPTN', 'OPTT', 'OPXS', 'OPY',
                   'OR', 'ORA', 'ORAN', 'ORC', 'ORCC', 'ORCL', 'ORGN', 'ORGO', 'ORGS', 'ORI', 'ORIA', 'ORIC', 'ORLA',
                   'ORLY', 'ORMP', 'ORN', 'ORRF', 'ORTX', 'OSA', 'OSBC', 'OSCR', 'OSG', 'OSH', 'OSI', 'OSIS', 'OSK',
                   'OSPN', 'OSS', 'OST', 'OSTK', 'OSUR', 'OSW', 'OTEC', 'OTEX', 'OTIS', 'OTLK', 'OTLY', 'OTMO', 'OTRK',
                   'OTTR', 'OUST', 'OUT', 'OVBC', 'OVID', 'OVLY', 'OVV', 'OWL', 'OWLT', 'OXAC', 'OXBR', 'OXM', 'OXSQ',
                   'OXUS', 'OXY', 'OZ', 'OZK', 'PAA', 'PAAS', 'PAC', 'PACB', 'PACI', 'PACK', 'PACW', 'PAG', 'PAGP',
                   'PAGS', 'PAHC', 'PALI', 'PALT', 'PAM', 'PANA', 'PANL', 'PANW', 'PAR', 'PARA', 'PARAA', 'PARR',
                   'PASG', 'PATH', 'PATI', 'PATK', 'PAVM', 'PAVS', 'PAX', 'PAY', 'PAYC', 'PAYO', 'PAYS', 'PAYX', 'PB',
                   'PBA', 'PBAX', 'PBBK', 'PBF', 'PBFS', 'PBH', 'PBHC', 'PBI', 'PBLA', 'PBPB', 'PBR', 'PBR.A', 'PBT',
                   'PBTS', 'PBYI', 'PCAR', 'PCB', 'PCCT', 'PCG', 'PCH', 'PCOR', 'PCRX', 'PCSA', 'PCT', 'PCTI', 'PCTY',
                   'PCVX', 'PCYG', 'PCYO', 'PD', 'PDCE', 'PDCO', 'PDD', 'PDEX', 'PDFS', 'PDLB', 'PDM', 'PDS', 'PDSB',
                   'PEAK', 'PEAR', 'PEB', 'PEBK', 'PEBO', 'PECO', 'PED', 'PEG', 'PEGA', 'PEGR', 'PEGY', 'PEN', 'PENN',
                   'PEP', 'PEPG', 'PEPL', 'PERF', 'PERI', 'PESI', 'PET', 'PETQ', 'PETS', 'PETV', 'PETZ', 'PEV', 'PFBC',
                   'PFC', 'PFE', 'PFG', 'PFGC', 'PFIE', 'PFIN', 'PFIS', 'PFLT', 'PFMT', 'PFS', 'PFSI', 'PFSW', 'PFTA',
                   'PFX', 'PG', 'PGC', 'PGEN', 'PGNY', 'PGR', 'PGRE', 'PGRU', 'PGRW', 'PGSS', 'PGTI', 'PGY', 'PH',
                   'PHAR', 'PHAT', 'PHCF', 'PHG', 'PHGE', 'PHI', 'PHIO', 'PHM', 'PHR', 'PHUN', 'PHVS', 'PHX', 'PHYT',
                   'PI', 'PIAI', 'PII', 'PIII', 'PIK', 'PINC', 'PINE', 'PINS', 'PIPR', 'PIRS', 'PIXY', 'PJT', 'PK',
                   'PKBK', 'PKE', 'PKG', 'PKI', 'PKOH', 'PKX', 'PL', 'PLAB', 'PLAG', 'PLAO', 'PLAY', 'PLBC', 'PLBY',
                   'PLCE', 'PLD', 'PLG', 'PLL', 'PLM', 'PLMI', 'PLMR', 'PLNT', 'PLOW', 'PLPC', 'PLRX', 'PLSE', 'PLTK',
                   'PLTN', 'PLTR', 'PLUG', 'PLUR', 'PLUS', 'PLX', 'PLXP', 'PLXS', 'PLYA', 'PLYM', 'PM', 'PMCB', 'PMD',
                   'PME', 'PMGM', 'PMN', 'PMT', 'PMTS', 'PMVP', 'PNAC', 'PNBK', 'PNC', 'PNFP', 'PNM', 'PNNT', 'PNR',
                   'PNRG', 'PNT', 'PNTG', 'PNTM', 'PNW', 'POAI', 'PODD', 'POET', 'POL', 'POLA', 'POOL', 'POR', 'PORT',
                   'POST', 'POWI', 'POWL', 'POWW', 'PPBI', 'PPBT', 'PPC', 'PPG', 'PPHP', 'PPIH', 'PPL', 'PPSI', 'PPTA',
                   'PPYA', 'PR', 'PRA', 'PRAA', 'PRAX', 'PRCH', 'PRCT', 'PRDO', 'PRDS', 'PRE', 'PRFT', 'PRFX', 'PRG',
                   'PRGO', 'PRGS', 'PRI', 'PRIM', 'PRK', 'PRLB', 'PRLD', 'PRLH', 'PRM', 'PRME', 'PRMW', 'PRO', 'PROC',
                   'PROF', 'PROK', 'PROV', 'PRPC', 'PRPH', 'PRPL', 'PRPO', 'PRQR', 'PRSO', 'PRSR', 'PRST', 'PRT',
                   'PRTA', 'PRTC', 'PRTG', 'PRTH', 'PRTK', 'PRTS', 'PRU', 'PRVA', 'PRVB', 'PSA', 'PSEC', 'PSFE', 'PSHG',
                   'PSMT', 'PSN', 'PSNL', 'PSNY', 'PSO', 'PSPC', 'PSTG', 'PSTL', 'PSTV', 'PSTX', 'PSX', 'PT', 'PTC',
                   'PTCT', 'PTE', 'PTEN', 'PTGX', 'PTHR', 'PTIX', 'PTLO', 'PTMN', 'PTN', 'PTON', 'PTPI', 'PTRA', 'PTRS',
                   'PTSI', 'PTVE', 'PTWO', 'PUBM', 'PUCK', 'PUK', 'PULM', 'PUMP', 'PUYI', 'PVBC', 'PVH', 'PVL', 'PW',
                   'PWFL', 'PWOD', 'PWP', 'PWR', 'PWSC', 'PWUP', 'PX', 'PXD', 'PXLW', 'PXMD', 'PXS', 'PYCR', 'PYPD',
                   'PYPL', 'PYR', 'PYXS', 'PZG', 'PZZA', 'QBTS', 'QCOM', 'QCRH', 'QD', 'QDEL', 'QDRO', 'QFIN', 'QFTA',
                   'QGEN', 'QH', 'QIPT', 'QLGN', 'QLI', 'QLYS', 'QMCO', 'QNCX', 'QNRX', 'QNST', 'QOMO', 'QRHC', 'QRTEA',
                   'QRTEB', 'QRVO', 'QS', 'QSG', 'QSI', 'QSR', 'QTEK', 'QTRX', 'QTWO', 'QUAD', 'QUBT', 'QUIK', 'QUOT',
                   'QURE', 'R', 'RAAS', 'RACE', 'RACY', 'RAD', 'RADI', 'RAIL', 'RAIN', 'RAM', 'RAMP', 'RAND', 'RANI',
                   'RAPT', 'RARE', 'RAVE', 'RAYA', 'RBA', 'RBB', 'RBBN', 'RBC', 'RBCAA', 'RBKB', 'RBLX', 'RBOT', 'RBT',
                   'RC', 'RCAC', 'RCAT', 'RCEL', 'RCFA', 'RCI', 'RCKT', 'RCKY', 'RCL', 'RCLF', 'RCM', 'RCMT', 'RCON',
                   'RCRT', 'RCUS', 'RDCM', 'RDFN', 'RDHL', 'RDI', 'RDIB', 'RDN', 'RDNT', 'RDVT', 'RDW', 'RDWR', 'RDY',
                   'RE', 'REAL', 'REAX', 'REBN', 'REE', 'REFI', 'REFR', 'REG', 'REGN', 'REI', 'REKR', 'RELI', 'RELL',
                   'RELX', 'RELY', 'RENE', 'RENN', 'RENT', 'REPL', 'REPX', 'RERE', 'RES', 'RETA', 'RETO', 'REUN',
                   'REVB', 'REVE', 'REVG', 'REX', 'REXR', 'REYN', 'REZI', 'RF', 'RFAC', 'RFIL', 'RFL', 'RGA', 'RGC',
                   'RGCO', 'RGEN', 'RGF', 'RGLD', 'RGLS', 'RGNX', 'RGP', 'RGR', 'RGS', 'RGTI', 'RH', 'RHE', 'RHI',
                   'RHP', 'RIBT', 'RICK', 'RIDE', 'RIG', 'RIGL', 'RILY', 'RIO', 'RIOT', 'RITM', 'RIVN', 'RJAC', 'RJF',
                   'RKDA', 'RKLB', 'RKT', 'RKTA', 'RL', 'RLAY', 'RLGT', 'RLI', 'RLJ', 'RLMD', 'RLX', 'RLYB', 'RM',
                   'RMAX', 'RMBI', 'RMBL', 'RMBS', 'RMCF', 'RMD', 'RMED', 'RMGC', 'RMNI', 'RMR', 'RMTI', 'RNA', 'RNAZ',
                   'RNG', 'RNGR', 'RNLX', 'RNR', 'RNST', 'RNW', 'RNXT', 'ROAD', 'ROC', 'ROCC', 'ROCG', 'ROCK', 'ROCL',
                   'ROG', 'ROIC', 'ROIV', 'ROK', 'ROKU', 'ROL', 'RONI', 'ROOT', 'ROP', 'ROSE', 'ROSS', 'ROST', 'ROVR',
                   'RPAY', 'RPD', 'RPHM', 'RPID', 'RPM', 'RPRX', 'RPT', 'RPTX', 'RRAC', 'RRBI', 'RRC', 'RRGB', 'RRR',
                   'RRX', 'RS', 'RSG', 'RSI', 'RSKD', 'RSLS', 'RSSS', 'RSVR', 'RTC', 'RTL', 'RTO', 'RTX', 'RUM', 'RUN',
                   'RUSHA', 'RUSHB', 'RUTH', 'RVLP', 'RVLV', 'RVMD', 'RVNC', 'RVP', 'RVPH', 'RVSB', 'RVSN', 'RVYL',
                   'RWAY', 'RWLK', 'RWOD', 'RWT', 'RXDX', 'RXO', 'RXRX', 'RXST', 'RXT', 'RY', 'RYAAY', 'RYAM', 'RYAN',
                   'RYI', 'RYN', 'RYTM', 'RZLT', 'S', 'SA', 'SABR', 'SABS', 'SACH', 'SAFE', 'SAFT', 'SAGA', 'SAGE',
                   'SAH', 'SAI', 'SAIA', 'SAIC', 'SAL', 'SALM', 'SAM', 'SAMA', 'SAMG', 'SAN', 'SANA', 'SAND', 'SANG',
                   'SANM', 'SANW', 'SAP', 'SAR', 'SASI', 'SASR', 'SATL', 'SATS', 'SATX', 'SAVA', 'SAVE', 'SB', 'SBAC',
                   'SBCF', 'SBET', 'SBEV', 'SBFG', 'SBFM', 'SBGI', 'SBH', 'SBIG', 'SBLK', 'SBOW', 'SBR', 'SBRA', 'SBS',
                   'SBSI', 'SBSW', 'SBT', 'SBUX', 'SBXC', 'SCAQ', 'SCCO', 'SCHL', 'SCHN', 'SCHW', 'SCI', 'SCKT', 'SCL',
                   'SCLX', 'SCM', 'SCOR', 'SCPH', 'SCPL', 'SCRM', 'SCS', 'SCSC', 'SCTL', 'SCU', 'SCUA', 'SCVL', 'SCWO',
                   'SCWX', 'SCX', 'SCYX', 'SD', 'SDAC', 'SDC', 'SDGR', 'SDIG', 'SDPI', 'SDRL', 'SE', 'SEAC', 'SEAS',
                   'SEAT', 'SEB', 'SECO', 'SEDA', 'SEDG', 'SEE', 'SEED', 'SEEL', 'SEER', 'SEIC', 'SELB', 'SELF', 'SEM',
                   'SEMR', 'SENEA', 'SENEB', 'SENS', 'SEPA', 'SERA', 'SES', 'SEV', 'SEVN', 'SF', 'SFBC', 'SFBS', 'SFE',
                   'SFIX', 'SFL', 'SFM', 'SFNC', 'SFR', 'SFST', 'SFT', 'SG', 'SGA', 'SGBX', 'SGC', 'SGEN', 'SGFY',
                   'SGH', 'SGHC', 'SGHL', 'SGHT', 'SGII', 'SGLY', 'SGMA', 'SGML', 'SGMO', 'SGRP', 'SGRY', 'SGTX', 'SGU',
                   'SHAK', 'SHAP', 'SHBI', 'SHC', 'SHCO', 'SHCR', 'SHEL', 'SHEN', 'SHFS', 'SHG', 'SHIP', 'SHLS', 'SHO',
                   'SHOO', 'SHOP', 'SHPH', 'SHPW', 'SHUA', 'SHW', 'SHYF', 'SI', 'SIBN', 'SID', 'SIDU', 'SIEB', 'SIEN',
                   'SIF', 'SIFY', 'SIG', 'SIGA', 'SIGI', 'SII', 'SILC', 'SILK', 'SILO', 'SILV', 'SIM', 'SIMO', 'SINT',
                   'SIRE', 'SIRI', 'SISI', 'SITC', 'SITE', 'SITM', 'SIX', 'SJ', 'SJM', 'SJR', 'SJT', 'SJW', 'SKE',
                   'SKGR', 'SKIL', 'SKIN', 'SKLZ', 'SKM', 'SKT', 'SKWD', 'SKX', 'SKY', 'SKYA', 'SKYH', 'SKYT', 'SKYW',
                   'SKYX', 'SLAB', 'SLAC', 'SLAM', 'SLB', 'SLCA', 'SLDB', 'SLDP', 'SLF', 'SLG', 'SLGC', 'SLGG', 'SLGL',
                   'SLGN', 'SLI', 'SLM', 'SLN', 'SLNA', 'SLND', 'SLNG', 'SLNH', 'SLNO', 'SLP', 'SLQT', 'SLRC', 'SLRX',
                   'SLS', 'SLVM', 'SLVR', 'SM', 'SMAP', 'SMAR', 'SMBC', 'SMBK', 'SMCI', 'SMFG', 'SMFL', 'SMG', 'SMHI',
                   'SMID', 'SMLP', 'SMLR', 'SMMF', 'SMMT', 'SMP', 'SMPL', 'SMR', 'SMRT', 'SMSI', 'SMTC', 'SMTI', 'SMWB',
                   'SMX', 'SNA', 'SNAL', 'SNAP', 'SNAX', 'SNBR', 'SNCE', 'SNCR', 'SNCY', 'SND', 'SNDA', 'SNDL', 'SNDR',
                   'SNDX', 'SNES', 'SNEX', 'SNFCA', 'SNGX', 'SNMP', 'SNN', 'SNOA', 'SNOW', 'SNPO', 'SNPS', 'SNPX',
                   'SNRH', 'SNSE', 'SNT', 'SNTG', 'SNTI', 'SNV', 'SNX', 'SNY', 'SO', 'SOBR', 'SOFI', 'SOFO', 'SOHO',
                   'SOHU', 'SOI', 'SOL', 'SOLO', 'SON', 'SOND', 'SONM', 'SONN', 'SONO', 'SONX', 'SONY', 'SOPA', 'SOPH',
                   'SOS', 'SOTK', 'SOUN', 'SOVO', 'SP', 'SPB', 'SPCB', 'SPCE', 'SPCM', 'SPFI', 'SPG', 'SPGI', 'SPH',
                   'SPI', 'SPIR', 'SPKB', 'SPLK', 'SPLP', 'SPNS', 'SPNT', 'SPOK', 'SPOT', 'SPPI', 'SPR', 'SPRB', 'SPRC',
                   'SPRO', 'SPRU', 'SPRY', 'SPSC', 'SPT', 'SPTN', 'SPWH', 'SPWR', 'SPXC', 'SQ', 'SQFT', 'SQL', 'SQM',
                   'SQNS', 'SQSP', 'SQZ', 'SR', 'SRAD', 'SRC', 'SRCE', 'SRCL', 'SRDX', 'SRE', 'SRG', 'SRGA', 'SRI',
                   'SRL', 'SRPT', 'SRRK', 'SRT', 'SRTS', 'SRZN', 'SSB', 'SSBI', 'SSBK', 'SSD', 'SSIC', 'SSKN', 'SSL',
                   'SSNC', 'SSNT', 'SSP', 'SSRM', 'SSSS', 'SST', 'SSTI', 'SSTK', 'SSU', 'SSY', 'SSYS', 'ST', 'STAA',
                   'STAF', 'STAG', 'STAR', 'STBA', 'STBX', 'STC', 'STCN', 'STE', 'STEL', 'STEM', 'STEP', 'STER', 'STET',
                   'STG', 'STGW', 'STIM', 'STIX', 'STKH', 'STKL', 'STKS', 'STLA', 'STLD', 'STM', 'STN', 'STNE', 'STNG',
                   'STOK', 'STR', 'STRA', 'STRC', 'STRE', 'STRL', 'STRM', 'STRO', 'STRR', 'STRS', 'STRT', 'STRW',
                   'STSA', 'STSS', 'STT', 'STTK', 'STVN', 'STWD', 'STX', 'STXS', 'STZ', 'SU', 'SUAC', 'SUI', 'SUM',
                   'SUMO', 'SUN', 'SUNL', 'SUNW', 'SUP', 'SUPN', 'SUPV', 'SURF', 'SURG', 'SUZ', 'SVC', 'SVFD', 'SVII',
                   'SVM', 'SVNA', 'SVRA', 'SVRE', 'SVT', 'SVVC', 'SWAG', 'SWAV', 'SWBI', 'SWI', 'SWIM', 'SWK', 'SWKH',
                   'SWKS', 'SWN', 'SWSS', 'SWTX', 'SWVL', 'SWX', 'SXC', 'SXI', 'SXT', 'SXTC', 'SY', 'SYBT', 'SYBX',
                   'SYF', 'SYK', 'SYM', 'SYNA', 'SYNH', 'SYPR', 'SYRS', 'SYTA', 'SYY', 'SZZL', 'T', 'TA', 'TAC', 'TACT',
                   'TAIT', 'TAK', 'TAL', 'TALK', 'TALO', 'TALS', 'TANH', 'TAOP', 'TAP', 'TARA', 'TARO', 'TARS', 'TASK',
                   'TAST', 'TATT', 'TAYD', 'TBBK', 'TBCP', 'TBI', 'TBIO', 'TBLA', 'TBLD', 'TBLT', 'TBNK', 'TBPH', 'TC',
                   'TCBC', 'TCBI', 'TCBK', 'TCBP', 'TCBS', 'TCBX', 'TCFC', 'TCI', 'TCMD', 'TCN', 'TCOA', 'TCOM', 'TCON',
                   'TCPC', 'TCRR', 'TCRT', 'TCRX', 'TCS', 'TCVA', 'TCX', 'TD', 'TDC', 'TDCX', 'TDG', 'TDOC', 'TDS',
                   'TDUP', 'TDW', 'TDY', 'TEAM', 'TECH', 'TECK', 'TEDU', 'TEF', 'TEL', 'TELA', 'TELL', 'TENB', 'TENK',
                   'TENX', 'TEO', 'TER', 'TERN', 'TESS', 'TETC', 'TETE', 'TEVA', 'TEX', 'TFC', 'TFFP', 'TFII', 'TFIN',
                   'TFPM', 'TFSL', 'TFX', 'TG', 'TGAA', 'TGAN', 'TGB', 'TGH', 'TGI', 'TGL', 'TGLS', 'TGNA', 'TGR',
                   'TGS', 'TGT', 'TGTX', 'TGVC', 'TH', 'THC', 'THCH', 'THCP', 'THFF', 'THG', 'THM', 'THMO', 'THO',
                   'THR', 'THRD', 'THRM', 'THRN', 'THRX', 'THRY', 'THS', 'THTX', 'TIG', 'TIGO', 'TIGR', 'TIL', 'TILE',
                   'TIMB', 'TIO', 'TIOA', 'TIPT', 'TIRX', 'TISI', 'TITN', 'TIVC', 'TIXT', 'TJX', 'TK', 'TKAT', 'TKC',
                   'TKLF', 'TKNO', 'TKR', 'TLF', 'TLGA', 'TLGY', 'TLIS', 'TLK', 'TLRY', 'TLS', 'TLSA', 'TLYS', 'TM',
                   'TMBR', 'TMC', 'TMCI', 'TMDX', 'TME', 'TMHC', 'TMKR', 'TMO', 'TMP', 'TMPO', 'TMQ', 'TMST', 'TMUS',
                   'TNC', 'TNDM', 'TNET', 'TNGX', 'TNK', 'TNL', 'TNON', 'TNP', 'TNXP', 'TNYA', 'TOAC', 'TOI', 'TOL',
                   'TOMZ', 'TOP', 'TOPS', 'TORO', 'TOST', 'TOUR', 'TOVX', 'TOWN', 'TPB', 'TPC', 'TPG', 'TPH', 'TPHS',
                   'TPIC', 'TPL', 'TPR', 'TPST', 'TPVG', 'TPX', 'TR', 'TRAQ', 'TRC', 'TRCA', 'TRDA', 'TREE', 'TREX',
                   'TRGP', 'TRHC', 'TRI', 'TRIB', 'TRIN', 'TRIP', 'TRIS', 'TRKA', 'TRMB', 'TRMD', 'TRMK', 'TRMR', 'TRN',
                   'TRNO', 'TRNS', 'TRON', 'TROO', 'TROW', 'TROX', 'TRP', 'TRS', 'TRST', 'TRT', 'TRTL', 'TRTN', 'TRTX',
                   'TRU', 'TRUE', 'TRUP', 'TRV', 'TRVG', 'TRVI', 'TRVN', 'TRX', 'TS', 'TSAT', 'TSBK', 'TSCO', 'TSE',
                   'TSEM', 'TSHA', 'TSLA', 'TSLX', 'TSM', 'TSN', 'TSP', 'TSQ', 'TSRI', 'TSVT', 'TT', 'TTC', 'TTCF',
                   'TTD', 'TTE', 'TTEC', 'TTEK', 'TTGT', 'TTI', 'TTMI', 'TTNP', 'TTOO', 'TTSH', 'TTWO', 'TU', 'TUP',
                   'TURN', 'TUSK', 'TUYA', 'TV', 'TVTX', 'TW', 'TWCB', 'TWI', 'TWIN', 'TWKS', 'TWLO', 'TWLV', 'TWNI',
                   'TWNK', 'TWO', 'TWOA', 'TWOU', 'TWST', 'TX', 'TXG', 'TXMD', 'TXN', 'TXO', 'TXRH', 'TXT', 'TYDE',
                   'TYG', 'TYL', 'TYRA', 'TZOO', 'U', 'UA', 'UAA', 'UAL', 'UAMY', 'UAN', 'UAVS', 'UBA', 'UBCP', 'UBER',
                   'UBFO', 'UBOH', 'UBP', 'UBS', 'UBSI', 'UBX', 'UCBI', 'UCL', 'UCTT', 'UDMY', 'UDR', 'UE', 'UEC',
                   'UEIC', 'UFAB', 'UFCS', 'UFI', 'UFPI', 'UFPT', 'UG', 'UGI', 'UGP', 'UGRO', 'UHAL', 'UHS', 'UHT',
                   'UI', 'UIHC', 'UIS', 'UK', 'UL', 'ULBI', 'ULCC', 'ULH', 'ULTA', 'UMBF', 'UMC', 'UMH', 'UNAM', 'UNB',
                   'UNCY', 'UNF', 'UNFI', 'UNH', 'UNIT', 'UNM', 'UNP', 'UNTY', 'UNVR', 'UONE', 'UONEK', 'UP', 'UPBD',
                   'UPC', 'UPH', 'UPLD', 'UPS', 'UPST', 'UPTD', 'UPWK', 'UPXI', 'URBN', 'URG', 'URGN', 'URI', 'UROY',
                   'USAC', 'USAP', 'USAS', 'USAU', 'USB', 'USCB', 'USCT', 'USDP', 'USEA', 'USEG', 'USFD', 'USIO',
                   'USLM', 'USM', 'USNA', 'USPH', 'USX', 'UTAA', 'UTHR', 'UTI', 'UTL', 'UTMD', 'UTME', 'UTRS', 'UTSI',
                   'UTZ', 'UUU', 'UUUU', 'UVE', 'UVSP', 'UVV', 'UWMC', 'UXIN', 'V', 'VABK', 'VAC', 'VACC', 'VAL',
                   'VALE', 'VALN', 'VALU', 'VANI', 'VAPO', 'VAQC', 'VATE', 'VAXX', 'VBFC', 'VBIV', 'VBLT', 'VBNK',
                   'VBOC', 'VBTX', 'VC', 'VCEL', 'VCNX', 'VCSA', 'VCTR', 'VCXA', 'VCXB', 'VCYT', 'VECO', 'VECT', 'VEDU',
                   'VEEE', 'VEEV', 'VEL', 'VEON', 'VERA', 'VERB', 'VERI', 'VERO', 'VERU', 'VERV', 'VERX', 'VERY', 'VET',
                   'VEV', 'VFC', 'VFF', 'VGAS', 'VGR', 'VGZ', 'VHAQ', 'VHC', 'VHI', 'VHNA', 'VIA', 'VIAO', 'VIAV',
                   'VICI', 'VICR', 'VIEW', 'VIGL', 'VII', 'VINC', 'VINE', 'VINO', 'VINP', 'VIOT', 'VIPS', 'VIR', 'VIRC',
                   'VIRI', 'VIRT', 'VIRX', 'VISL', 'VIST', 'VITL', 'VIV', 'VIVK', 'VJET', 'VKTX', 'VLAT', 'VLCN', 'VLD',
                   'VLGEA', 'VLN', 'VLO', 'VLON', 'VLRS', 'VLTA', 'VLY', 'VMAR', 'VMC', 'VMCA', 'VMD', 'VMEO', 'VMGA',
                   'VMI', 'VMW', 'VNCE', 'VNDA', 'VNET', 'VNO', 'VNOM', 'VNRX', 'VNT', 'VNTR', 'VOC', 'VOD', 'VOR',
                   'VORB', 'VOXR', 'VOXX', 'VOYA', 'VPG', 'VQS', 'VRA', 'VRAR', 'VRAX', 'VRAY', 'VRCA', 'VRDN', 'VRE',
                   'VREX', 'VRM', 'VRME', 'VRNA', 'VRNS', 'VRNT', 'VRPX', 'VRRM', 'VRSK', 'VRSN', 'VRT', 'VRTS', 'VRTV',
                   'VRTX', 'VS', 'VSAC', 'VSAT', 'VSCO', 'VSEC', 'VSH', 'VST', 'VSTA', 'VSTM', 'VSTO', 'VTEX', 'VTGN',
                   'VTLE', 'VTNR', 'VTOL', 'VTR', 'VTRS', 'VTRU', 'VTS', 'VTSI', 'VTVT', 'VTYX', 'VUZI', 'VVI', 'VVOS',
                   'VVPR', 'VVV', 'VVX', 'VWE', 'VXRT', 'VYGR', 'VYNE', 'VYNT', 'VZ', 'VZIO', 'VZLA', 'W', 'WAB',
                   'WABC', 'WAFD', 'WAFU', 'WAL', 'WALD', 'WASH', 'WAT', 'WATT', 'WAVC', 'WAVD', 'WAVE', 'WAVS', 'WB',
                   'WBA', 'WBD', 'WBS', 'WBX', 'WCC', 'WCN', 'WD', 'WDAY', 'WDC', 'WDFC', 'WDH', 'WDS', 'WE', 'WEAV',
                   'WEC', 'WEJO', 'WEL', 'WELL', 'WEN', 'WERN', 'WES', 'WEST', 'WETG', 'WEX', 'WEYS', 'WF', 'WFC',
                   'WFCF', 'WFG', 'WFRD', 'WGO', 'WGS', 'WH', 'WHD', 'WHF', 'WHG', 'WHLM', 'WHLR', 'WHR', 'WILC',
                   'WIMI', 'WINA', 'WING', 'WINT', 'WINV', 'WIRE', 'WISA', 'WISH', 'WIT', 'WIX', 'WK', 'WKEY', 'WKHS',
                   'WKME', 'WKSP', 'WLDN', 'WLDS', 'WLFC', 'WLK', 'WLKP', 'WLMS', 'WLY', 'WLYB', 'WM', 'WMB', 'WMC',
                   'WMG', 'WMK', 'WMPN', 'WMS', 'WMT', 'WNC', 'WNEB', 'WNNR', 'WNS', 'WNW', 'WOLF', 'WOOF', 'WOR',
                   'WORX', 'WOW', 'WPC', 'WPM', 'WPP', 'WPRT', 'WRAC', 'WRAP', 'WRB', 'WRBY', 'WRK', 'WRLD', 'WRN',
                   'WSBC', 'WSBF', 'WSC', 'WSFS', 'WSM', 'WSO', 'WSR', 'WST', 'WT', 'WTBA', 'WTER', 'WTFC', 'WTI',
                   'WTM', 'WTMA', 'WTRG', 'WTS', 'WTT', 'WTTR', 'WTW', 'WU', 'WULF', 'WVE', 'WVVI', 'WW', 'WWAC', 'WWD',
                   'WWE', 'WWR', 'WWW', 'WY', 'WYNN', 'WYY', 'X', 'XAIR', 'XBIO', 'XBIT', 'XCUR', 'XEL', 'XELA', 'XELB',
                   'XENE', 'XERS', 'XFIN', 'XFOR', 'XGN', 'XHR', 'XIN', 'XLO', 'XM', 'XMTR', 'XNCR', 'XNET', 'XOM',
                   'XOMA', 'XOS', 'XP', 'XPAX', 'XPDB', 'XPEL', 'XPER', 'XPEV', 'XPL', 'XPO', 'XPOF', 'XPON', 'XPRO',
                   'XRAY', 'XRTX', 'XRX', 'XTLB', 'XTNT', 'XWEL', 'XXII', 'XYF', 'XYL', 'YALA', 'YCBD', 'YELL', 'YELP',
                   'YETI', 'YEXT', 'YGMZ', 'YI', 'YJ', 'YMAB', 'YMM', 'YORW', 'YOSH', 'YOTA', 'YOU', 'YPF', 'YQ', 'YRD',
                   'YS', 'YSG', 'YTEN', 'YTPG', 'YTRA', 'YUM', 'YUMC', 'YVR', 'YY', 'Z', 'ZBH', 'ZBRA', 'ZCMD', 'ZD',
                   'ZDGE', 'ZENV', 'ZEPP', 'ZETA', 'ZEUS', 'ZEV', 'ZFOX', 'ZG', 'ZGN', 'ZH', 'ZI', 'ZIM', 'ZIMV',
                   'ZING', 'ZION', 'ZIP', 'ZIVO', 'ZKIN', 'ZLAB', 'ZM', 'ZNTL', 'ZOM', 'ZS', 'ZT', 'ZTEK', 'ZTO', 'ZTS',
                   'ZUMZ', 'ZUO', 'ZURA', 'ZVIA', 'ZVRA', 'ZVSA', 'ZWS', 'ZYME', 'ZYNE', 'ZYXI']
        volumes = []
        for name in tqdm(tickers):
            stock = yf.Ticker(name)
            hist = stock.history(period='max')
            volumes.append(hist.Volume.to_list())

        volumes = [[v for v in volume if v] for volume in volumes if volume and len(volume) >= 100]

        with open('experiments/stocks.pkl', 'wb') as fp:
            pickle.dump(volumes, fp)

    return volumes


if __name__ == '__main__':
    main()
