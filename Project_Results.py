import matplotlib.pyplot as plt
import re
from collections import defaultdict
import io

# The content of your result_1.txt file is stored in this multiline string.
# If your file changes, you can update the content inside the triple quotes.
sampling_100_100_10_4_16 = """
100-100(Sampling, 10_4_16)

100-100[None, 0, 10 infer, 4 seq, 16 prl][10:29<00:00,  4.92s/it]
  test/2opt_iterations             5.625
      test/gt_cost           7.738855388272474
  test/merge_iterations       912.9580078125
    test/solved_cost        7.7392806227371755

100-100[Christofides_c, 800, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.93s/it]
  test/2opt_iterations           5.671875
      test/gt_cost           7.738855388272474
  test/merge_iterations       965.3740234375
    test/solved_cost         7.739129341546931

100-100[Christofides_c, 750, 10 infer, 4 seq, 16 prl][10:30<00:00,  4.92s/it]
  test/2opt_iterations           5.609375
      test/gt_cost           7.738855388272474
  test/merge_iterations         916.9453125
    test/solved_cost         7.739198528009898

100-100[Christofides_c, 700, 10 infer, 4 seq, 16 prl][10:30<00:00,  4.93s/it]
  test/2opt_iterations           5.7578125
      test/gt_cost           7.738855388272474
  test/merge_iterations          993.53125
    test/solved_cost         7.739242114334026

100-100[Christofides_c, 650, 10 infer, 4 seq, 16 prl][10:30<00:00,  4.93s/it]
  test/2opt_iterations            5.71875
      test/gt_cost           7.738855388272474
  test/merge_iterations        903.27734375
    test/solved_cost         7.739105513924753

100-100[Christofides_c, 500, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.93s/it]
  test/2opt_iterations           5.8359375
      test/gt_cost           7.738855388272474
  test/merge_iterations         985.140625
    test/solved_cost        7.7393729071121315

100-100[nearest_neighbor_c, 800, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.94s/it]
  test/2opt_iterations           5.8984375
      test/gt_cost           7.738855388272474
  test/merge_iterations         1049.59375
    test/solved_cost         7.739106226788904

100-100[nearest_neighbor_c, 750, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.93s/it]
  test/2opt_iterations            5.96875
      test/gt_cost           7.738855388272474
  test/merge_iterations       948.7939453125
    test/solved_cost         7.739142008909745

100-100[nearest_neighbor_c, 700, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.93s/it]
  test/2opt_iterations           5.7578125
      test/gt_cost           7.738855388272474
  test/merge_iterations        973.60546875
    test/solved_cost         7.739162226811696

100-100[nearest_neighbor_c, 650, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.93s/it]
  test/2opt_iterations            5.65625
      test/gt_cost           7.738855388272474
  test/merge_iterations       901.5087890625
    test/solved_cost         7.739350261911644

100-100[nearest_neighbor_c, 500, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.94s/it]
  test/2opt_iterations           5.6328125
      test/gt_cost           7.738855388272474
  test/merge_iterations       945.5302734375
    test/solved_cost         7.739042449539218

100-100[convex_hull_c, 800, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.94s/it]
  test/2opt_iterations           5.671875
      test/gt_cost           7.738855388272474
  test/merge_iterations       963.4775390625
    test/solved_cost         7.739020743219612

100-100[convex_hull_c, 750, 10 infer, 4 seq, 16 prl][10:30<00:00,  4.93s/it]
  test/2opt_iterations            5.53125
      test/gt_cost           7.738855388272474
  test/merge_iterations        932.87890625
    test/solved_cost         7.739344152614413

100-100[convex_hull_c, 700, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.93s/it]
  test/2opt_iterations            5.8125
      test/gt_cost           7.738855388272474
  test/merge_iterations         946.3359375
    test/solved_cost         7.739309637358266

100-100[convex_hull_c, 650, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.93s/it]
  test/2opt_iterations           5.6171875
      test/gt_cost           7.738855388272474
  test/merge_iterations       884.3525390625
    test/solved_cost         7.739265354673252

100-100[convex_hull_c, 500, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.93s/it]
  test/2opt_iterations            5.65625
      test/gt_cost           7.738855388272474
  test/merge_iterations        1010.87109375
    test/solved_cost         7.739174561886218

100-100[farthest_c, 800, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.93s/it]
  test/2opt_iterations           5.7265625
      test/gt_cost           7.738855388272474
  test/merge_iterations        951.810546875
    test/solved_cost         7.739308713407469

100-100[farthest_c, 750, 10 infer, 4 seq, 16 prl][10:30<00:00,  4.93s/it]
  test/2opt_iterations           5.6015625
      test/gt_cost           7.738855388272474
  test/merge_iterations        925.021484375
    test/solved_cost         7.739034946229732

100-100[farthest_c, 700, 10 infer, 4 seq, 16 prl][10:30<00:00,  4.92s/it]
  test/2opt_iterations           5.6484375
      test/gt_cost           7.738855388272474
  test/merge_iterations        943.669921875
    test/solved_cost        7.7392792521376075

100-100[farthest_c, 650, 10 infer, 4 seq, 16 prl][10:31<00:00,  4.93s/it]
  test/2opt_iterations           5.6171875
      test/gt_cost           7.738855388272474
  test/merge_iterations           906.875
    test/solved_cost         7.739096422357678

100-100[farthest_c, 500, 10 infer, 4 seq, 16 prl][10:30<00:00,  4.93s/it]
  test/2opt_iterations           5.6640625
      test/gt_cost           7.738855388272474
  test/merge_iterations       930.7197265625
    test/solved_cost         7.73918316530971
"""

file_100_100_greedy = """

100-100(Greedy)

100-100[None, 0, 50 infer, 4 seq, 1 prl][04:01<00:00,  1.89s/it]
  test/2opt_iterations           1.3359375
      test/gt_cost           7.738855388272474
  test/merge_iterations          796.65625
    test/solved_cost        7.7423798494615195

100-100[Christofides_c, 500, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.90s/it]
  test/2opt_iterations           1.515625
      test/gt_cost           7.738855388272474
  test/merge_iterations          902.90625
    test/solved_cost         7.744895102709798

100-100[Christofides_c, 550, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.90s/it]
  test/2opt_iterations            1.53125
      test/gt_cost           7.738855388272474
  test/merge_iterations         799.453125
    test/solved_cost         7.742041438443282

100-100[Christofides_c, 600, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.90s/it]
  test/2opt_iterations           1.2421875
      test/gt_cost           7.738855388272474
  test/merge_iterations          666.15625
    test/solved_cost         7.74231111966097

100-100[Christofides_c, 650, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.90s/it]
  test/2opt_iterations            1.34375
      test/gt_cost           7.738855388272474
  test/merge_iterations         881.671875
    test/solved_cost         7.742171611747047

100-100[Christofides_c, 700, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.90s/it]
  test/2opt_iterations            1.53125
      test/gt_cost           7.738855388272474
  test/merge_iterations         778.296875
    test/solved_cost         7.743863026507673

100-100[Christofides_c, 750, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.90s/it]
  test/2opt_iterations             1.75
      test/gt_cost           7.738855388272474
  test/merge_iterations          913.46875
    test/solved_cost         7.743389936990241

100-100[Christofides_c, 800, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.90s/it]
  test/2opt_iterations           1.4296875
      test/gt_cost           7.738855388272474
  test/merge_iterations          659.65625
    test/solved_cost         7.743456381587578

100-100[nearest_neighbor_c, 500, 50 infer, 4 seq, 1 prl][04:03<00:00,  1.90s/it]
  test/2opt_iterations           1.609375
      test/gt_cost           7.738855388272474
  test/merge_iterations         966.015625
    test/solved_cost        7.7429240306609675

100-100[nearest_neighbor_c, 550, 50 infer, 4 seq, 1 prl][04:03<00:00,  1.90s/it]
  test/2opt_iterations            1.53125
      test/gt_cost           7.738855388272474
  test/merge_iterations         838.046875
    test/solved_cost         7.741821048779814

100-100[nearest_neighbor_c, 600, 50 infer, 4 seq, 1 prl][04:03<00:00,  1.90s/it]
  test/2opt_iterations           1.703125
      test/gt_cost           7.738855388272474
  test/merge_iterations         916.890625
    test/solved_cost         7.742141876170405

100-100[nearest_neighbor_c, 650, 50 infer, 4 seq, 1 prl][04:03<00:00,  1.90s/it]
   test/2opt_iterations           1.328125
      test/gt_cost           7.738855388272474
  test/merge_iterations         640.296875
    test/solved_cost         7.741664218830544

100-100[nearest_neighbor_c, 700, 50 infer, 4 seq, 1 prl][04:03<00:00,  1.90s/it]
  test/2opt_iterations           1.3203125
      test/gt_cost           7.738855388272474
  test/merge_iterations         817.140625
    test/solved_cost        7.7414658382853405

100-100[nearest_neighbor_c, 750, 50 infer, 4 seq, 1 prl][04:03<00:00,  1.90s/it]
  test/2opt_iterations           1.359375
      test/gt_cost           7.738855388272474
  test/merge_iterations         673.703125
    test/solved_cost         7.743234710229528

100-100[nearest_neighbor_c, 800, 50 infer, 4 seq, 1 prl][04:03<00:00,  1.90s/it]
  test/2opt_iterations            1.15625
      test/gt_cost           7.738855388272474
  test/merge_iterations         740.546875
    test/solved_cost         7.743334980538554

100-100[convex_hull_c, 500, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.90s/it]
  test/2opt_iterations           1.3984375
      test/gt_cost           7.738855388272474
  test/merge_iterations           688.625
    test/solved_cost         7.741654247915656

100-100[convex_hull_c, 550, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.90s/it]
  test/2opt_iterations           1.453125
      test/gt_cost           7.738855388272474
  test/merge_iterations         652.796875
    test/solved_cost         7.742700533272803

100-100[convex_hull_c, 600, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.90s/it]
  test/2opt_iterations             1.375
      test/gt_cost           7.738855388272474
  test/merge_iterations         750.859375
    test/solved_cost         7.743000996115518

100-100[convex_hull_c, 650, 50 infer, 4 seq, 1 prl][04:03<00:00,  1.90s/it]
  test/2opt_iterations           1.3203125
      test/gt_cost           7.738855388272474
  test/merge_iterations          698.9375
    test/solved_cost         7.742064412896826

100-100[convex_hull_c, 700, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.89s/it]
  test/2opt_iterations           1.2421875
      test/gt_cost           7.738855388272474
  test/merge_iterations           512.75
    test/solved_cost         7.742319228335684

100-100[convex_hull_c, 750, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.89s/it]
  test/2opt_iterations           1.0546875
      test/gt_cost           7.738855388272474
  test/merge_iterations         502.234375
    test/solved_cost         7.741870979175384

100-100[convex_hull_c, 800, 50 infer, 4 seq, 1 prl][04:03<00:00,  1.90s/it]
  test/2opt_iterations           1.8515625
      test/gt_cost           7.738855388272474
  test/merge_iterations          988.84375
    test/solved_cost         7.742621078998794

100-100[farthest_c, 500, 50 infer, 4 seq, 1 prl][04:01<00:00,  1.89s/it]
  test/2opt_iterations              1.5
      test/gt_cost           7.738855388272474
  test/merge_iterations         716.359375
    test/solved_cost        7.7424214433880305

100-100[farthest_c, 550, 50 infer, 4 seq, 1 prl][04:01<00:00,  1.89s/it]
  test/2opt_iterations            1.5625
      test/gt_cost           7.738855388272474
  test/merge_iterations         949.140625
    test/solved_cost         7.742865635888046

100-100[farthest_c, 600, 50 infer, 4 seq, 1 prl][04:01<00:00,  1.89s/it]
  test/2opt_iterations           1.578125
      test/gt_cost           7.738855388272474
  test/merge_iterations          958.8125
    test/solved_cost         7.742159605483319

100-100[farthest_c, 650, 50 infer, 4 seq, 1 prl][04:01<00:00,  1.89s/it]
  test/2opt_iterations             1.375
      test/gt_cost           7.738855388272474
  test/merge_iterations         660.796875
    test/solved_cost         7.741593769645563

100-100[farthest_c, 700, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.89s/it]
  test/2opt_iterations           1.5546875
      test/gt_cost           7.738855388272474
  test/merge_iterations         987.296875
    test/solved_cost         7.742472291354879

100-100[farthest_c, 750, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.89s/it]
  test/2opt_iterations           1.421875
      test/gt_cost           7.738855388272474
  test/merge_iterations         756.859375
    test/solved_cost        7.7421260806289185

100-100[farthest_c, 800, 50 infer, 4 seq, 1 prl][04:02<00:00,  1.89s/it]
  test/2opt_iterations            1.3125
      test/gt_cost           7.738855388272474
  test/merge_iterations         750.421875
    test/solved_cost         7.742174668308993 """

sampling_100_100_10_4_8 = """
100-100(Sampling)(10, 4, 8)

100-100[None, 0, 10 infer, 4 seq, 8 prl][05:21<00:00,  2.51s/it]
  test/2opt_iterations           5.1171875
      test/gt_cost           7.738855388272474
  test/merge_iterations        964.044921875
    test/solved_cost         7.739427006689671

100-100[Christofides_c, 800, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations            4.84375
      test/gt_cost           7.738855388272474
  test/merge_iterations        955.220703125
    test/solved_cost        7.7393647276495035

100-100[Christofides_c, 750, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations            4.78125
      test/gt_cost           7.738855388272474
  test/merge_iterations       1001.404296875
    test/solved_cost         7.739405245101381

100-100[Christofides_c, 700, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.921875
      test/gt_cost           7.738855388272474
  test/merge_iterations       1043.498046875
    test/solved_cost         7.739370496019297

100-100[Christofides_c, 650, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.6484375
      test/gt_cost           7.738855388272474
  test/merge_iterations        912.373046875
    test/solved_cost         7.739385506194278

100-100[Christofides_c, 600, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.7734375
      test/gt_cost           7.738855388272474
  test/merge_iterations       1038.787109375
    test/solved_cost         7.739367995499621

100-100[Christofides_c, 550, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.5546875
      test/gt_cost           7.738855388272474
  test/merge_iterations        940.365234375
    test/solved_cost         7.739486803539888

100-100[Christofides_c, 500, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.6171875
      test/gt_cost           7.738855388272474
  test/merge_iterations            889.5
    test/solved_cost         7.739439572596916

100-100[nearest_neighbor_c, 800, 10 infer, 4 seq, 8 prl][05:23<00:00,  2.53s/it]
  test/2opt_iterations           4.515625
      test/gt_cost           7.738855388272474
  test/merge_iterations        917.451171875
    test/solved_cost         7.739450806201267

100-100[nearest_neighbor_c, 750, 10 infer, 4 seq, 8 prl][05:23<00:00,  2.53s/it]
  test/2opt_iterations            4.5625
      test/gt_cost           7.738855388272474
  test/merge_iterations       1002.060546875
    test/solved_cost         7.73928777499099

100-100[nearest_neighbor_c, 700, 10 infer, 4 seq, 8 prl][05:23<00:00,  2.53s/it]
  test/2opt_iterations           4.578125
      test/gt_cost           7.738855388272474
  test/merge_iterations        900.412109375
    test/solved_cost         7.739476753914003

100-100[nearest_neighbor_c, 650, 10 infer, 4 seq, 8 prl][05:23<00:00,  2.53s/it]
  test/2opt_iterations           4.578125
      test/gt_cost           7.738855388272474
  test/merge_iterations        996.103515625
    test/solved_cost         7.739482244163221

100-100[nearest_neighbor_c, 600, 10 infer, 4 seq, 8 prl][05:23<00:00,  2.53s/it]
  test/2opt_iterations           4.8671875
      test/gt_cost           7.738855388272474
  test/merge_iterations       1027.583984375
    test/solved_cost         7.739336300179564

100-100[nearest_neighbor_c, 550, 10 infer, 4 seq, 8 prl][05:23<00:00,  2.53s/it]
  test/2opt_iterations           4.7421875
      test/gt_cost           7.738855388272474
  test/merge_iterations         968.515625
    test/solved_cost        7.7393828668524165

100-100[nearest_neighbor_c, 500, 10 infer, 4 seq, 8 prl][05:23<00:00,  2.53s/it]
  test/2opt_iterations           4.9296875
      test/gt_cost           7.738855388272474
  test/merge_iterations         981.7890625
    test/solved_cost         7.739354911167542

100-100[convex_hull_c, 800, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.2109375
      test/gt_cost           7.738855388272474
  test/merge_iterations        910.755859375
    test/solved_cost         7.739576932385501

100-100[convex_hull_c, 750, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.5390625
      test/gt_cost           7.738855388272474
  test/merge_iterations            955.5
    test/solved_cost         7.739491968891335

100-100[convex_hull_c, 700, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.8671875
      test/gt_cost           7.738855388272474
  test/merge_iterations           967.25
    test/solved_cost         7.739474338369107

100-100[convex_hull_c, 650, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations              4.5
      test/gt_cost           7.738855388272474
  test/merge_iterations        959.326171875
    test/solved_cost         7.739563652417922

100-100[convex_hull_c, 600, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations            4.6875
      test/gt_cost           7.738855388272474
  test/merge_iterations         900.7578125
    test/solved_cost        7.7393787785556185

100-100[convex_hull_c, 550, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations            4.96875
      test/gt_cost           7.738855388272474
  test/merge_iterations        923.33984375
    test/solved_cost         7.739335736963953

100-100[convex_hull_c, 500, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.8828125
      test/gt_cost           7.738855388272474
  test/merge_iterations          964.65625
    test/solved_cost         7.739391799990634

100-100[farthest_c, 800, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations            4.71875
      test/gt_cost           7.738855388272474
  test/merge_iterations        920.619140625
    test/solved_cost         7.739440303673036

100-100[farthest_c, 750, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.5078125
      test/gt_cost           7.738855388272474
  test/merge_iterations        869.90234375
    test/solved_cost         7.739349884911922

100-100[farthest_c, 700, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations            4.90625
      test/gt_cost           7.738855388272474
  test/merge_iterations       1001.080078125
    test/solved_cost         7.739616912044182

100-100[farthest_c, 650, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.4140625
      test/gt_cost           7.738855388272474
  test/merge_iterations        904.787109375
    test/solved_cost         7.739404717818927

100-100[farthest_c, 600, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations            4.96875
      test/gt_cost           7.738855388272474
  test/merge_iterations        1067.37890625
    test/solved_cost         7.73948672705121

100-100[farthest_c, 550, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.9609375
      test/gt_cost           7.738855388272474
  test/merge_iterations        944.376953125
    test/solved_cost        7.7394301436388915

100-100[farthest_c, 500, 10 infer, 4 seq, 8 prl][05:22<00:00,  2.52s/it]
  test/2opt_iterations           4.484375
      test/gt_cost           7.738855388272474
  test/merge_iterations         880.890625
    test/solved_cost         7.739705555344212

"""


def parse_and_plot_results(data):
    """
    Parses the experiment results from a string, calculates the gap,
    and plots the results.

    Args:
        data (str): A multiline string containing the log file content.
    """
    # Use a defaultdict to easily append data
    results = defaultdict(list)
    
    # Use io.StringIO to read the string data line by line
    data_stream = io.StringIO(data)
    lines = data_stream.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Find the line that contains the algorithm and noise info
        if line.startswith("100-100["):
            try:
                # Extract content within the first bracket
                match_content = line.split('[')[1].split(']')[0]
                parts = [p.strip() for p in match_content.split(',')]
                
                algorithm = parts[0]
                noise = int(parts[1])

                # Look for gt_cost and solved_cost in the next few lines
                gt_cost, solved_cost = None, None
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("100-100["):
                    if "test/gt_cost" in lines[j]:
                        gt_cost = float(lines[j].split()[-1])
                    if "test/solved_cost" in lines[j]:
                        solved_cost = float(lines[j].split()[-1])
                    j += 1
                
                if gt_cost is not None and solved_cost is not None:
                    # Calculate the Gap (%)
                    gap = (solved_cost - gt_cost) / gt_cost * 100
                    results[algorithm].append((noise, gap))
                
                i = j -1 # Move index to the start of the next block
            except (IndexError, ValueError) as e:
                # This handles cases where parsing might fail for a specific line
                print(f"Skipping line due to parsing error: {line} -> {e}")
                pass
        i += 1

    # Plotting the results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot baseline 'None' as a horizontal line if it exists
    if 'None' in results:
        none_data = results.pop('None')
        if none_data:
            # Baseline is at noise=0, but we plot it across the x-axis for comparison
            ax.axhline(y=none_data[0][1], color='k', linestyle='--', label=f'Baseline (None-Guided): {none_data[0][1]:.6f}%')

    # Plot each algorithm's data
    for algorithm, data_points in sorted(results.items()):
        # Sort points by noise level for a clean line plot
        data_points.sort()
        noises = [dp[0] for dp in data_points]
        gaps = [dp[1] for dp in data_points]
        if algorithm == "Christofides_c":
            # Highlight Christofides_c with a different color
            ax.plot(noises, gaps, marker='o', linestyle='-', label="Christofides", color='blue')
        elif algorithm == "nearest_neighbor_c":
            # Highlight nearest_neighbor_c with a different color
            ax.plot(noises, gaps, marker='o', linestyle='-', label="Nearest Neighbor", color='red')
        elif algorithm == "convex_hull_c":
            # Highlight convex_hull_c with a different color
            ax.plot(noises, gaps, marker='o', linestyle='-', label="Convex hull+Insertion", color='green')
        elif algorithm == "farthest_c":
            # Highlight farthest_c with a different color
            ax.plot(noises, gaps, marker='o', linestyle='-', label="Farthest Insertion", color='orange')
    
    # --- Chart Customization ---
    ax.set_title('Guidance Algorithm Performance(100 nodes, Greedy)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Noise', fontsize=12)
    ax.set_ylabel('Gap (%)', fontsize=12)
    
    # Invert x-axis as per the typical representation for this kind of data
    ax.invert_xaxis()
    
    # Add a legend to distinguish the lines
    ax.legend(title='Guidance Algorithm', fontsize=10)
    
    # Improve tick readability
    plt.xticks(rotation=45)
    plt.tight_layout() # Adjust layout to make room for labels
    
    # Show the plot
    plt.show()


# --- Main execution ---
if __name__ == "__main__":
    parse_and_plot_results(data=file_100_100_greedy)
    #parse_and_plot_results(data=sampling_100_100_10_4_16)
    #parse_and_plot_results(data=sampling_100_100_10_4_8)

