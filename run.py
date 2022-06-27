from dataset import get_dataset
from helpers import *
from info_model import *
from loop import Loop
from model_injection import ModelInjection
from plotter import Plotter
from max_model import *
from typing import Callable, List
# from scalene import scalene_profiler

IMAGENET_IMGS_PATH = '../loop-injection/ILSVRC2012_img_val/'
IMAGENET_LABELS_PATH = '../loop-injection/LOC_val_solution.csv'

'''
BELOW ARE 100 IMAGES (ImageNet images) CORRECTLY IDENTIFIED BY NETWORKS
'''
sample_alexnet_correct_img_inds =  [3082, 158, 14632, 16904, 23404, 49554, 27420, 10, 40355, 46009, 
                                    13675, 29498, 36621, 5011, 36865, 32133, 46302, 23568, 34584, 
                                    36341, 6809, 7659, 47723, 9513, 1361, 27938, 32900, 32495, 25318, 
                                    15760, 8804, 18470, 14098, 20520, 21095, 42497, 45825, 246, 45807, 
                                    49895, 9613, 44234, 12282, 30678, 16030, 5993, 1002, 16595, 15806, 
                                    53, 19063, 2906, 18922, 18422, 34743, 40722, 47097, 10045, 28411, 
                                    8923, 10338, 13078, 42032, 25365, 2750, 32989, 11514, 17934, 9527, 
                                    12764, 35337, 30304, 29563, 4785, 14316, 35586, 15234, 15413, 41321, 
                                    34803, 42140, 3866, 12202, 45605, 28266, 6239, 42078, 28040, 38439, 
                                    6170, 31110, 25479, 9357, 45000, 44803, 16733, 41813, 48290, 28367, 23290]

sample_resnet18_correct_img_inds =  [47954, 15766, 27187, 11948, 39609, 19792, 4785, 7539, 39025, 27848, 
                                     21369, 30689, 19290, 17066, 37888, 29550, 35999, 38688, 14589, 17802, 
                                     20755, 24738, 1515, 45277, 24869, 38481, 20884, 19879, 30945, 26961, 
                                     49454, 10504, 2082, 19615, 5791, 23980, 22481, 26090, 42121, 22937, 
                                     39033, 14041, 33218, 40055, 6033, 44883, 36531, 25603, 32308, 40989, 
                                     38478, 22163, 40921, 15540, 12577, 13161, 26162, 21088, 30717, 41688, 
                                     12317, 17581, 26933, 43763, 38480, 37230, 36331, 36472, 18034, 22320, 
                                     31481, 2936, 16301, 30080, 45712, 47908, 29129, 16199, 46379, 21498, 
                                     37792, 14990, 25322, 49822, 48683, 12616, 36544, 8044, 30432, 16955, 
                                     23151, 16317, 49390, 48113, 49691, 15180, 773, 18106, 35598, 28573]

sample_efficientnet_b0_correct_img_inds = [15787, 20235, 20137, 21675, 27337, 23977, 20436, 14359, 48530, 39141, 
                                           40903, 33434, 45043, 24367, 16851, 18142, 25887, 29461, 8453, 28723, 
                                           13095, 20968, 28635, 1088, 39024, 39888, 37172, 24266, 22750, 45712, 
                                           40611, 19038, 6238, 42345, 12727, 4120, 32875, 16233, 38111, 46479, 
                                           38113, 5850, 37596, 22216, 34970, 25542, 35972, 27099, 8554, 23113, 
                                           15304, 41906, 32020, 28962, 2070, 15103, 1550, 23495, 6950, 18046, 
                                           14867, 28084, 34787, 14090, 45125, 41515, 8874, 26732, 4249, 27487, 
                                           28724, 40538, 42365, 49274, 19159, 13276, 23167, 23100, 171, 20022, 
                                           40319, 4847, 49086, 10716, 5404, 28912, 2287, 46950, 8175, 38458, 
                                           33044, 45250, 3032, 47744, 38126, 24312, 22692, 9570, 14887, 42331]

sample_vit_224_correct_img_inds = [37998, 4455, 40687, 11835, 48774, 19823, 41053, 6596, 2569, 33822, 
                                   16641, 34773, 33809, 36362, 29372, 2313, 46457, 44544, 1322, 8487, 
                                   33764, 45344, 983, 16531, 5561, 25032, 24852, 44568, 17543, 12206, 
                                   46905, 28573, 15891, 21499, 6538, 43889, 39222, 21495, 12770, 10374, 
                                   41707, 43216, 18416, 34008, 48110, 40433, 11048, 18136, 20299, 1276, 
                                   30043, 36987, 40412, 29932, 48807, 15501, 39310, 19603, 15783, 1954, 
                                   30187, 26781, 18634, 22682, 7441, 46886, 25257, 44924, 42824, 46812, 
                                   1922, 24601, 8818, 28567, 612, 3291, 37775, 7912, 25268, 14645, 9105, 
                                   23607, 5696, 22532, 33620, 27205, 47491, 12069, 34515, 36954, 29617, 
                                   11427, 41341, 38234, 40647, 2401, 24612, 21436, 24681, 39629]

sample_deit_tiny_correct_img_inds = [36488, 31453, 40026, 3227, 19750, 40403, 42648, 1485, 39040, 41095, 
                                     8752, 19126, 14532, 45820, 16889, 3721, 13890, 20200, 23161, 32472, 
                                     49590, 23700, 23534, 29270, 23060, 48867, 33546, 33643, 33454, 8612, 
                                     27986, 2242, 37716, 47661, 157, 8685, 39036, 39551, 10728, 32112, 
                                     6618, 29750, 12863, 28519, 20945, 47499, 39766, 7913, 30547, 37958, 
                                     17420, 16839, 24063, 35025, 35258, 4017, 49828, 36191, 7348, 11925, 
                                     362, 12195, 23443, 28555, 49580, 16151, 23145, 30962, 17312, 35184, 
                                     41476, 41, 46034, 41527, 17251, 48876, 42940, 41506, 12456, 1516, 
                                     31301, 43036, 2017, 11709, 10878, 21374, 14693, 34548, 5024, 6781, 
                                     49158, 46768, 12518, 43404, 49288, 38353, 39658, 38423, 18951, 40490]


'''
BELOW ARE THE MAXES AND MINS FOR CONV/FC LAYERS OF EACH NETWORK
'''
# all taken over 1000 trials of randomly chosen images
ALEXNET_MAX = [60.63791275024414, 153.95208740234375, 157.78179931640625, 114.32186126708984, 59.255977630615234, 53.963340759277344, 50.961307525634766, 50.21159362792969]
ALEXNET_MIN = [-64.5768814086914, -287.6980285644531, -304.3017272949219, -232.31802368164062, -115.81592559814453, -81.17062377929688, -83.81184387207031, -17.84949493408203]

RESNET18_MAX = [36.13761901855469, 10.887314796447754, 3.855881929397583, 9.794642448425293, 3.2933764457702637, 8.875794410705566, 4.089364528656006, 4.49278450012207, 5.494577884674072, 3.7413206100463867, 5.697751522064209, 3.7465898990631104, 1.6894534826278687, 4.060337543487549, 2.0673913955688477, 3.928053855895996, 2.009352684020996, 2.0958352088928223, 3.4922521114349365, 3.5720021724700928, 39.165164947509766]
RESNET18_MIN = [-35.18002700805664, -16.452085494995117, -7.513538837432861, -11.784904479980469, -5.422745704650879, -9.174695014953613, -3.548478841781616, -5.52264404296875, -5.273807048797607, -3.147308349609375, -6.394559383392334, -4.2612624168396, -1.897513508796692, -5.157448768615723, -2.5855915546417236, -5.034069538116455, -1.9327064752578735, -1.9150633811950684, -5.472661018371582, -0.9709599614143372, -13.11787223815918]

MOBILENETV3_MAX = [50.529754638671875, 152.4453125, 0.0, 1.3555821180343628, 16.386011123657227, 78.19126892089844, 71.61649322509766, 33.39183807373047, 61.84816360473633, 8.23404312133789, 15.759119987487793, 35.79911804199219, 29.254331588745117, 2.615121364593506, 2.923299789428711, 13.406920433044434, 11.035303115844727, 6.359800338745117, 4.915771961212158, 4.819554805755615, 2.160830497741699, 13.238816261291504, 9.492086410522461, 3.6886425018310547, 3.8758585453033447, 4.501008987426758, 37.39836502075195, 29.129955291748047, 6.472546577453613, 7.442508697509766, 16.180858612060547, 17.564950942993164, 8.270240783691406, 3.4738659858703613, 3.808992862701416, 2.436999559402466, 19.86147689819336, 10.726669311523438, 4.361920356750488, 5.574491500854492, 6.010217189788818, 10.493760108947754, 6.449452877044678, 5.642043590545654, 11.042641639709473, 3.060288906097412, 14.884574890136719, 5.985340595245361, 9.629993438720703, 13.182779312133789, 7.347139835357666, 43.87862014770508, 8.678768157958984, 35.42095184326172]
MOBILENETV3_MIN = [-46.409305572509766, -36.219173431396484, -0.022064918652176857, -1.137220025062561, -18.50782012939453, -89.23754119873047, -51.432743072509766, -78.7419204711914, -111.61312103271484, -11.161054611206055, -10.881126403808594, -59.42047119140625, -53.47572326660156, -25.13960075378418, -3.0235414505004883, -17.344646453857422, -11.549857139587402, -9.487520217895508, -12.789621353149414, -9.113523483276367, -2.441288709640503, -16.640043258666992, -7.28016996383667, -4.473825931549072, -6.601798057556152, -3.565438747406006, -39.14116287231445, -17.95857048034668, -15.7798490524292, -8.16700267791748, -17.298330307006836, -12.253621101379395, -6.461938858032227, -2.318554639816284, -4.474216938018799, -2.78330659866333, -16.941946029663086, -19.40937614440918, -4.8647942543029785, -6.048542499542236, -6.270298004150391, -12.721344947814941, -9.488436698913574, -3.6828505992889404, -8.396136283874512, -3.970151662826538, -20.726463317871094, -7.242910861968994, -6.34951114654541, -8.538702011108398, -9.124135971069336, -33.122108459472656, -16.190759658813477, -30.60342788696289]

EFFICIENTNET_B0_MAX = [41.261878967285156, 93.59493255615234, 0.0, 4.2753005027771, 45.14156723022461, 166.43971252441406, 103.35803985595703, 6.432144641876221, 4.794723033905029, 110.14116668701172, 108.8333969116211, 15.859593391418457, 6.081498146057129, 5.799123764038086, 22.822816848754883, 142.0337371826172, 26.79233741760254, 9.409040451049805, 6.848724842071533, 31.58502197265625, 66.91728210449219, 14.487445831298828, 6.489365100860596, 5.050395965576172, 7.100586414337158, 131.93551635742188, 11.453241348266602, 1.2970048189163208, 3.4181692600250244, 16.753894805908203, 77.97586822509766, 10.88070011138916, 12.38310432434082, 5.333405017852783, 3.4936363697052, 67.88749694824219, 6.922603130340576, 9.3286771774292, 5.355068683624268, 2.3240394592285156, 109.18562316894531, 16.67705535888672, 7.414438247680664, 5.535353660583496, 18.822690963745117, 71.87299346923828, 10.00155258178711, 14.929365158081055, 10.084174156188965, 6.788357734680176, 68.50354766845703, 11.664961814880371, 13.255234718322754, 6.019690990447998, 5.377971172332764, 114.84893798828125, 7.1525492668151855, 16.387714385986328, 6.658006191253662, 28.52964973449707, 79.24200439453125, 6.458648204803467, 22.334917068481445, 9.318705558776855, 16.27557373046875, 92.03357696533203, 11.54696273803711, 22.59390640258789, 8.029184341430664, 12.47804069519043, 159.3617401123047, 8.038491249084473, 30.680871963500977, 12.82648754119873, 12.533780097961426, 138.0638427734375, 11.33388614654541, 22.70612335205078, 36.37251281738281, 16.552732467651367, 47.85399627685547, 16.57033920288086]
EFFICIENTNET_B0_MIN = [-41.29035186767578, -205.56234741210938, -17.28680419921875, -2.0289430618286133, -32.14414978027344, -274.0902099609375, -105.8837890625, -12.400992393493652, -1.865220308303833, -158.46310424804688, -168.59085083007812, -26.46217918395996, -8.404044151306152, -4.349876403808594, -20.109745025634766, -156.60781860351562, -42.11426544189453, -29.866008758544922, -5.196669578552246, -39.086631774902344, -73.9615478515625, -23.468761444091797, -7.718266487121582, -7.501905918121338, -6.033256530761719, -121.68380737304688, -18.314838409423828, -9.487512588500977, -1.6473759412765503, -22.29636001586914, -65.77313232421875, -10.316733360290527, -16.215778350830078, -12.375143051147461, -5.417911529541016, -83.02943420410156, -13.467045783996582, -17.560989379882812, -7.312125205993652, -2.528489351272583, -105.98078155517578, -30.335445404052734, -5.013499736785889, -8.692607879638672, -24.284061431884766, -68.5529556274414, -8.454017639160156, -11.288421630859375, -11.162079811096191, -7.642269134521484, -74.83853912353516, -8.634333610534668, -5.772274017333984, -7.541411876678467, -6.518733501434326, -105.99881744384766, -18.082204818725586, -15.042597770690918, -8.613715171813965, -35.615875244140625, -64.91595458984375, -10.29785442352295, -10.506199836730957, -13.324463844299316, -14.760071754455566, -105.15850830078125, -10.621930122375488, -15.610891342163086, -17.292068481445312, -13.729647636413574, -128.69638061523438, -14.488265037536621, -19.056549072265625, -24.4056396484375, -11.058432579040527, -152.11376953125, -12.750208854675293, -20.739852905273438, -29.16045379638672, -17.621679306030273, -49.05451202392578, -6.7043867111206055]

VIT_224_MAX = [21.38619613647461, 13.137690544128418, 5.223912715911865, 22.591272354125977, 14.627334594726562, 8.52755355834961, 6.0006914138793945, 9.890716552734375, 10.100648880004883, 7.496172904968262, 2.838361978530884, 9.791748046875, 4.218319892883301, 7.627469062805176, 2.1772141456604004, 7.022871971130371, 9.97861385345459, 8.753116607666016, 2.603743314743042, 10.7113676071167, 9.364392280578613, 8.279913902282715, 1.913900375366211, 12.028547286987305, 16.298311233520508, 7.50547981262207, 3.8653643131256104, 11.191254615783691, 24.671464920043945, 8.172269821166992, 5.880016803741455, 8.650896072387695, 16.48027801513672, 9.161516189575195, 8.323050498962402, 21.478208541870117, 12.032465934753418, 9.890218734741211, 5.357192039489746, 39.167720794677734, 15.495792388916016, 10.776861190795898, 3.434782028198242, 82.16219329833984, 47.833740234375, 10.250325202941895, 7.804947376251221, 12.901052474975586, 27.16724395751953, 18.20071792602539]
VIT_224_MIN = [-19.463239669799805, -15.500661849975586, -5.93727970123291, -24.368146896362305, -10.279200553894043, -9.283302307128906, -6.6186628341674805, -15.092547416687012, -9.99910831451416, -8.01309871673584, -2.0055863857269287, -13.931312561035156, -6.891602039337158, -7.821361541748047, -1.683425784111023, -13.337827682495117, -20.036043167114258, -8.76114559173584, -2.5503268241882324, -16.220672607421875, -29.580533981323242, -8.281740188598633, -3.102416753768921, -14.465797424316406, -37.66347122192383, -8.265819549560547, -4.104371070861816, -9.637489318847656, -29.86809730529785, -8.226944923400879, -3.0855355262756348, -11.563165664672852, -7.577116966247559, -8.028410911560059, -5.703629493713379, -21.38522720336914, -5.2278337478637695, -9.873750686645508, -4.976212024688721, -45.72148513793945, -7.363213062286377, -9.736888885498047, -10.837485313415527, -176.64419555664062, -21.725860595703125, -11.022520065307617, -35.40918731689453, -22.09223175048828, -11.808673858642578, -6.43269157409668]

DEIT_TINY_MAX = [13.53121566772461, 7.818448066711426, 4.0968475341796875, 13.890387535095215, 9.439072608947754, 6.611987113952637, 3.497903347015381, 8.710073471069336, 2.9448904991149902, 7.335442543029785, 1.8504533767700195, 5.713381290435791, 2.0483291149139404, 5.946750164031982, 1.7645349502563477, 4.338800430297852, 1.8961845636367798, 5.5678205490112305, 1.6480587720870972, 4.394930839538574, 2.674673557281494, 5.509000301361084, 1.990847110748291, 6.493055820465088, 6.031866550445557, 5.592031478881836, 2.4142093658447266, 6.230477809906006, 4.256953716278076, 5.687406063079834, 1.8214855194091797, 6.867565631866455, 3.454225778579712, 5.7702317237854, 1.7727397680282593, 9.576885223388672, 4.66068172454834, 5.032649993896484, 1.5727978944778442, 8.312976837158203, 6.162326812744141, 5.1836676597595215, 1.5372108221054077, 6.127212047576904, 6.677909851074219, 5.323587417602539, 5.20518159866333, 6.405534744262695, 6.299277305603027, 12.445219039916992]
DEIT_TINY_MIN = [-15.050702095031738, -7.197841167449951, -4.5391011238098145, -14.679038047790527, -7.168591022491455, -6.879803657531738, -3.3527164459228516, -9.490851402282715, -3.0126757621765137, -7.761324882507324, -2.414383888244629, -10.00381851196289, -2.1800215244293213, -6.230242729187012, -1.9013686180114746, -8.018929481506348, -1.7346975803375244, -5.63113260269165, -1.5391839742660522, -11.119327545166016, -1.9761685132980347, -5.38472843170166, -1.4113953113555908, -9.08962631225586, -3.9388506412506104, -5.587961673736572, -1.6736162900924683, -8.546092987060547, -2.788851499557495, -5.3724870681762695, -1.7169537544250488, -9.444092750549316, -2.814786434173584, -6.113471031188965, -1.998713731765747, -15.362776756286621, -3.740246295928955, -5.539764881134033, -1.978590726852417, -12.448845863342285, -3.498164176940918, -5.06332540512085, -2.246274948120117, -10.400650024414062, -3.6200661659240723, -5.488716125488281, -2.930126905441284, -9.933143615722656, -4.7958784103393555, -6.523073673248291]


'''
BELOW ARE SAMPLED LAYERS FOR CERTAIN NETWORKS
'''
resnet18_sample_layers = [0, 1, 5, 6, 10, 13, 18, 19, 20]
mobilenet_sample_layers = [0, 1, 2, 6, 10, 12, 15, 18, 22, 28, 35, 45, 50, 51]
efficientnet_sample_layers = [0, 3, 5, 10, 13, 20, 30, 37, 46, 56, 60, 70, 80, 81]
vit_224_sample_layers = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48, 49]
deit_tiny_sample_layers = vit_224_sample_layers

def run_nvdla_inputs():
    # get the dataset and network
    dataset = get_dataset()
    
    num_layers, var_sizes, paddings, strides, _ = get_layer_info(get_alexnet, dataset[0]['image'])

    # get the loop objects
    loops = []

    # layer 1
    # 1: [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
    nvdla_vars_1 = [('m', 4), ('m', 16, True), ('c', 3), ('q', 11), ('p', 11), ('c', 1), 
                    ('s', 11), ('r', 11), ('q', 5), ('p', 5), ('r', 1), ('s', 1)]
    mem_dividers_1 = [0, 1, 10]
    nvdla_injection_1 = Loop(nvdla_vars_1, mem_dividers_1, d_type='i', sizes=var_sizes[0], paddings=paddings[0], input_strides=[4, 4])
    loops.append(nvdla_injection_1)

    # layer 2
    # 2: [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
    nvdla_vars_2 = [('m', 12), ('m', 16, True), ('c', 64), ('q', 8), ('p', 8), ('c', 1), 
                    ('s', 5), ('r', 5), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_2 = [0, 1, 10]
    nvdla_injection_2 = Loop(nvdla_vars_2, mem_dividers_2, d_type='i', sizes=var_sizes[1], paddings=paddings[1], input_strides=[1, 1])
    loops.append(nvdla_injection_2)

    # layer 3
    # 3: [(384, 192, 3, 3),    (1, 384, 13, 13),   (1, 192, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_3 = [('m', 24), ('m', 16, True), ('c', 192), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_3 = [0, 1, 10]
    nvdla_injection_3 = Loop(nvdla_vars_3, mem_dividers_3, d_type='i', sizes=var_sizes[2], paddings=paddings[2], input_strides=[1, 1])
    loops.append(nvdla_injection_3)

    # layer 4
    # 4: [(256, 384, 3, 3),    (1, 256, 13, 13),   (1, 384, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_4 = [('m', 16), ('m', 16, True), ('c', 384), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_4 = [0, 1, 10]
    nvdla_injection_4 = Loop(nvdla_vars_4, mem_dividers_4, d_type='i', sizes=var_sizes[3], paddings=paddings[3], input_strides=[1, 1])
    loops.append(nvdla_injection_4)

    # layer 5
    # 5: [(256, 256, 3, 3),    (1, 256, 13, 13),   (1, 256, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_5 = [('m', 16), ('m', 16, True), ('c', 256), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_5 = [0, 1, 10]
    nvdla_injection_5 = Loop(nvdla_vars_5, mem_dividers_5, d_type='i', sizes=var_sizes[4], paddings=paddings[4], input_strides=[1, 1])
    loops.append(nvdla_injection_5)

    mod_inj = ModelInjection(get_alexnet, dataset, 'alexnet', 'nvdla', loops, maxes=ALEXNET_MAX, mins=ALEXNET_MIN, overwrite=False, debug=True)
    correct_rate = mod_inj.full_inject(mode="bit", bit=5, img_inds=[], debug=True, inj_sites=[], layers=[])
    return correct_rate

def get_nvdla_loops(model_name:str, d_type:str):
    # get the dataset and network
    dataset = get_dataset()
    
    num_layers, var_sizes, paddings, strides, _ = get_layer_info(get_alexnet, dataset[0]['image'])
    
    if model_name == "alexnet":
        loops = []
        nvdla_vars = []
        # layer 1
        # 1: [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
        nvdla_vars_1 = [('m', 4), ('m', 16, True), ('c', 3), ('q', 11), ('p', 11), ('c', 1), 
                        ('s', 11), ('r', 11), ('q', 5), ('p', 5), ('r', 1), ('s', 1)]
        mem_dividers_1w = [0, 1, 10]
        nvdla_injection_1 = Loop(nvdla_vars_1, mem_dividers_1w, d_type=d_type, sizes=var_sizes[0], paddings=paddings[0], input_strides=[4, 4])
        loops.append(nvdla_injection_1)

        # layer 2
        # 2: [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
        nvdla_vars_2 = [('m', 12), ('m', 16, True), ('c', 64), ('q', 8), ('p', 8), ('c', 1), 
                        ('s', 5), ('r', 5), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
        mem_dividers_2w = [0, 1, 10]
        nvdla_injection_2 = Loop(nvdla_vars_2, mem_dividers_2w, d_type=d_type, sizes=var_sizes[1], paddings=paddings[1], input_strides=[1, 1])
        loops.append(nvdla_injection_2)

        # layer 3
        # 3: [(384, 192, 3, 3),    (1, 384, 13, 13),   (1, 192, 13, 13),   (1, 1),     (1, 1)]
        nvdla_vars_3 = [('m', 24), ('m', 16, True), ('c', 192), ('q', 4), ('p', 4), ('c', 1), 
                        ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
        mem_dividers_3w = [0, 1, 10]
        nvdla_injection_3 = Loop(nvdla_vars_3, mem_dividers_3w, d_type=d_type, sizes=var_sizes[2], paddings=paddings[2], input_strides=[1, 1])
        loops.append(nvdla_injection_3)

        # layer 4
        # 4: [(256, 384, 3, 3),    (1, 256, 13, 13),   (1, 384, 13, 13),   (1, 1),     (1, 1)]
        nvdla_vars_4 = [('m', 16), ('m', 16, True), ('c', 384), ('q', 4), ('p', 4), ('c', 1), 
                        ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
        mem_dividers_4w = [0, 1, 10]
        nvdla_injection_4 = Loop(nvdla_vars_4, mem_dividers_4w, d_type=d_type, sizes=var_sizes[3], paddings=paddings[3], input_strides=[1, 1])
        loops.append(nvdla_injection_4)

        # layer 5
        # 5: [(256, 256, 3, 3),    (1, 256, 13, 13),   (1, 256, 13, 13),   (1, 1),     (1, 1)]
        nvdla_vars_5 = [('m', 16), ('m', 16, True), ('c', 256), ('q', 4), ('p', 4), ('c', 1), 
                        ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
        mem_dividers_5w = [0, 1, 10]
        nvdla_injection_5 = Loop(nvdla_vars_5, mem_dividers_5w, d_type=d_type, sizes=var_sizes[4], paddings=paddings[4], input_strides=[1, 1])
        loops.append(nvdla_injection_5)
   
def run_nvdla_resnet18_inputs():
    # get the dataset and network
    dataset = get_dataset(IMAGENET_LABELS_PATH, IMAGENET_IMGS_PATH)
    d_type='i'
    
    num_layers, var_sizes, paddings, strides, _ = get_layer_info(get_resnet18, dataset[0]['image'])

    # get the loop objects
    loops = []
    
    def add_loop(loop_list, n):
        for i in range(n):
            loop_list.append(0)
        return loop_list
    
    # layer 0
    nvdla_vars = [('m', 4), ('m', 16, True), ('c', 3), ('q', 28), ('p', 28), ('c', 1), 
                    ('s', 7), ('r', 7), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[0], input_strides=strides[0])
    loops.append(nvdla_injection)
    
    # layer 1
    nvdla_vars = [('m', 4), ('m', 16, True), ('c', 64), ('q', 14), ('p', 14), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[1], input_strides=strides[1])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 3)
    
    # layer 5
    nvdla_vars = [('m', 8), ('m', 16, True), ('c', 64), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[5], input_strides=strides[5])
    loops.append(nvdla_injection)
    
    # layer 6
    nvdla_vars = [('m', 8), ('m', 16, True), ('c', 128), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[6], input_strides=strides[6])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 3)
    
    # layer 10
    nvdla_vars = [('m', 16), ('m', 16, True), ('c', 128), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 2), ('p', 2), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[10], input_strides=strides[10])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 2)
    
    # layer 13
    nvdla_vars = [('m', 16), ('m', 16, True), ('c', 256), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 2), ('p', 2), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[13], input_strides=strides[13])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 4)
    
    # layer 18
    nvdla_vars = [('m', 32), ('m', 16, True), ('c', 512), ('q', 1), ('p', 1), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 7), ('p', 7), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[18], input_strides=strides[18])
    loops.append(nvdla_injection)
    
    # layer 19
    nvdla_vars = [('m', 32), ('m', 16, True), ('c', 512), ('q', 1), ('p', 1), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 7), ('p', 7), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[19], input_strides=strides[19])
    loops.append(nvdla_injection)
    
    debug = True
    inj_inds = []
    layers = [0, 1, 5, 6, 10, 13, 18, 19]
    
    mod_inj = ModelInjection(get_resnet18, dataset, 'resnet18', 'nvdla', loops, maxes=RESNET18_MAX, mins=RESNET18_MIN, overwrite=False, debug=debug, d_type=d_type, max_range=True)
    correct_rate = mod_inj.full_inject(mode="bit", bit=range(1, 9), img_inds=sample_resnet18_correct_img_inds, debug=debug, inj_sites=inj_inds, layers=layers)
    
def run_nvdla_resnet18_weights():
    # get the dataset and network
    dataset = get_dataset(IMAGENET_LABELS_PATH, IMAGENET_IMGS_PATH)
    d_type='w'
    
    num_layers, var_sizes, paddings, strides, _ = get_layer_info(get_resnet18, dataset[0]['image'])

    # get the loop objects
    loops = []
    
    def add_loop(loop_list, n):
        for i in range(n):
            loop_list.append(0)
        return loop_list
    
    # layer 0
    nvdla_vars = [('m', 4), ('m', 16, True), ('c', 3), ('q', 28), ('p', 28), ('c', 1), 
                    ('s', 7), ('r', 7), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[0], input_strides=strides[0])
    loops.append(nvdla_injection)
    
    # layer 1
    nvdla_vars = [('m', 4), ('m', 16, True), ('c', 64), ('q', 14), ('p', 14), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[1], input_strides=strides[1])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 3)
    
    # layer 5
    nvdla_vars = [('m', 8), ('m', 16, True), ('c', 64), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[5], input_strides=strides[5])
    loops.append(nvdla_injection)
    
    # layer 6
    nvdla_vars = [('m', 8), ('m', 16, True), ('c', 128), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[6], input_strides=strides[6])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 3)
    
    # layer 10
    nvdla_vars = [('m', 16), ('m', 16, True), ('c', 128), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 2), ('p', 2), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[10], input_strides=strides[10])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 2)
    
    # layer 13
    nvdla_vars = [('m', 16), ('m', 16, True), ('c', 256), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 2), ('p', 2), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[13], input_strides=strides[13])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 4)
    
    # layer 18
    nvdla_vars = [('m', 32), ('m', 16, True), ('c', 512), ('q', 1), ('p', 1), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 7), ('p', 7), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[18], input_strides=strides[18])
    loops.append(nvdla_injection)
    
    # layer 19
    nvdla_vars = [('m', 32), ('m', 16, True), ('c', 512), ('q', 1), ('p', 1), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 7), ('p', 7), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[19], input_strides=strides[19])
    loops.append(nvdla_injection)
    
    debug = True
    inj_inds = []
    layers = [0, 1, 5, 6, 10, 13, 18, 19]
    
    mod_inj = ModelInjection(get_resnet18, dataset, 'resnet18', 'nvdla', loops, maxes=RESNET18_MAX, mins=RESNET18_MIN, overwrite=False, debug=debug, d_type=d_type, max_range=True)
    correct_rate = mod_inj.full_inject(mode="bit", bit=range(1, 9), img_inds=sample_resnet18_correct_img_inds, debug=debug, inj_sites=inj_inds, layers=layers)
    
def pick_maxmin(model_name:str):
    if model_name == "resnet18":
        return (RESNET18_MAX, RESNET18_MIN)
    elif model_name == "alexnet":
        return (ALEXNET_MAX, ALEXNET_MIN)
    elif model_name == "efficientnet_b0":
        return (EFFICIENTNET_B0_MAX, EFFICIENTNET_B0_MIN)
    elif model_name == "vit_224":
        return (VIT_224_MAX, VIT_224_MIN)
    elif model_name == "deit_tiny":
        return (DEIT_TINY_MAX, DEIT_TINY_MIN)
    else:
        return ([], [])
        
def pick_img_inds(model_name:str):
    if model_name == "resnet18":
        return sample_resnet18_correct_img_inds
    elif model_name == "alexnet":
        return sample_alexnet_correct_img_inds
    elif model_name == "efficientnet_b0":
        return sample_efficientnet_b0_correct_img_inds
    elif model_name == "vit_224":
        return sample_vit_224_correct_img_inds
    elif model_name == "deit_tiny":
        return sample_deit_tiny_correct_img_inds
    else:
        return []
    
def run_injection(get_net:Callable, model_name:str, arch_name:str, d_type:str="i", layers:List=[], inj_inds:List=[], loops:List=[], overwrite:bool=False, print_loops:bool=False):
    dataset = get_dataset(IMAGENET_LABELS_PATH, IMAGENET_IMGS_PATH)
    
    num_layers, var_sizes, paddings, strides, FC_sizes = get_layer_info(get_net, dataset[0]['image'])
    
    loops, names = get_loops(get_net, 'timeloop_maps/' + arch_name + '/' + model_name + '/', var_sizes, paddings, strides, d_type=d_type, layers=layers)
    if print_loops:
        for loop in loops:
            print(loop)
        assert(False)
    
    debug = True
    maxes, mins = pick_maxmin(model_name)
    
    mod_inj = ModelInjection(get_net, dataset, model_name, arch_name, loops, maxes=maxes, mins=mins, overwrite=overwrite, debug=debug, d_type=d_type, max_range=True)
    correct_rate = mod_inj.full_inject(mode="bit", bit=range(1, 9), img_inds=pick_img_inds(model_name), debug=debug, inj_sites=inj_inds, layers=layers)
    print(correct_rate)

def get_network_max(get_net, get_dataset, n=1000):
    net = get_net()
    dataset = get_dataset(csv_file=IMAGENET_LABELS_PATH, root_dir=IMAGENET_IMGS_PATH)
    maxes, mins = get_range(net, dataset, n=n)
    print(maxes)
    print(mins)
    
def get_resnet18_max_img(img):
    net = get_resnet18()
    maxes, mins = get_range_img(net, img)
    print(maxes)
    print(mins)

def run_plot(arch_name, net_name, add_on='', d_type='i', layers=[], correct_rate=1.0, xentropy=False, sparsity=False, num_sites=False, maxes_mins=False, sites_ratio=False, all=False):
    if arch_name == 'eyeriss':
        if d_type == 'i':
            mem_levels = ['DRAM', 'shared_glb', 'ifmap_spad']
        else:
            mem_levels = ['DRAM', 'weights_spad']
    elif arch_name == 'nvdla':
        if d_type == 'i':
            mem_levels = ['DRAM', 'GlobalBuffer', 'input_reg']
        else:
            mem_levels = ['DRAM', 'GlobalBuffer', 'weight_reg']
    elif arch_name == 'simba':
        if d_type == 'i':
            mem_levels = ['DRAM', 'GlobalBuffer', 'PEInputBuffer']
        else:
            mem_levels = ['DRAM', 'PEWeightBuffer', 'PEWeightRegs']
    else:
        assert(False and "I don't know this architecture")
        
    if maxes_mins or all:
        maxes, mins = pick_maxmin(net_name)
        new_maxes = []
        new_mins = []
        for i in range(len(maxes)):
            # if i in layers:
            new_maxes.append(maxes[i])
            new_mins.append(mins[i])
        maxmin = (new_maxes, new_mins)
    else:
        maxmin = []
        
    plotter = Plotter(arch_name, net_name, d_type=d_type, add_on=add_on, layers=layers)
    if all:
        num_options = len([xentropy, sparsity, num_sites, maxes_mins, sites_ratio])
        for i in range(num_options):
            if i==3:
                new_maxmin = maxmin
            else:
                new_maxmin = []
            plotter.plot_v2(level_names=mem_levels, xentropy=i==0, sparsity=i==1, num_sites=i==2, maxes_mins=new_maxmin, sites_ratio=i==4)
    else:
        plotter.plot_v2(level_names=mem_levels, xentropy=xentropy, sparsity=sparsity, num_sites=num_sites, maxes_mins=maxmin, sites_ratio=sites_ratio)

if __name__=="__main__":
    
    '''
    Prints a table with layer sizes and layer ids
    '''
    # print_layer_sizes(get_alexnet(), 'alexnet')
    
    '''
    Gets the maxes/mins of the outputs of the conv/FC layers of network provided
    '''
    # get_network_max(get_alexnet, get_dataset, n=10000)
    
    
    '''
    Automatically constructs the loops from given mappings and runs an injection experiment
    '''
    # run_injection(get_alexnet, "alexnet", "eyeriss", d_type="i")
    # run_injection(get_alexnet, "alexnet", "eyeriss", d_type="w")
    
    '''
    Plots the data collected from an injection experiment (in the data files)
    '''
    # run_plot('eyeriss', 'alexnet', d_type='i', add_on="")
    # run_plot('eyeriss', 'alexnet', d_type='w', add_on="")
    
    pass