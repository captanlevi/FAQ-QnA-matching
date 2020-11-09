from ..generation.rajat_work.qgen.generator.symsub import SymSubGenerator
from ..generation.rajat_work.qgen.generator.fpm.fpm import FPMGenerator
from ..generation.rajat_work.qgen.encoder.dummy import dummyEN
from ..generation.rajat_work.qgen.generator.eda import EDAGenerator
#from ..generation.broken_english.broken_english_generator import BrokenEnglishGen
from ..generation.sentAug.sentAug import AUG
from .generation import GenerateManager

defaultNames  = ["SymSub", "FPM", "EDA", "nlpAug"]
defaultNums = [2,4,4,2]

defaultGenerateManager = GenerateManager(producers= [SymSubGenerator(dummyEN("lite")),FPMGenerator(),EDAGenerator(), AUG()], names= defaultNames,nums= defaultNums)