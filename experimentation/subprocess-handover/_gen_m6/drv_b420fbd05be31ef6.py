def kern(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, c0, c1, c2, c3, c4, c5, b0, b1, b2, out):
    n = out.shape[0]
    for i in range(n):
        res = -1
        if res == -1:
            if (f13[i] == f13[i] and (c0[i] == 2 or c0[i] == 0) and c2[i] != 0 and c3[i] == 6):
                res = 757
            else:
                res = 228
        if res == -1:
            if (f6[i] == f6[i] and not b2[i]):
                if f9[i] < 0.8426:
                    if c4[i] == 2 or c4[i] == 3:
                        res = 898
                    elif c4[i] == 0 or c4[i] == 5:
                        res = 691
                    else:
                        res = 45
                elif f9[i] < 0.8703:
                    if c0[i] == 5 or c0[i] == 2:
                        res = 501
                    elif c0[i] == 9 or c0[i] == 3:
                        res = 705
                    elif c0[i] == 1 or c0[i] == 8:
                        res = 160
                    else:
                        res = 178
                else:
                    if c5[i] == 7 or c5[i] == 5:
                        res = 190
                    elif c5[i] == 2 or c5[i] == 3:
                        res = 636
                    elif c5[i] == 9 or c5[i] == 6:
                        res = 702
                    else:
                        res = 575
            else:
                if f6[i] < 0.7045:
                    res = 884
                else:
                    res = 448
        if res == -1:
            if (f3[i] > 0.5670 or (c4[i] == 8 or c4[i] == 4 or c4[i] == 5 or c4[i] == 9) or f13[i] != f13[i] or f7[i] > 0.2452):
                if (c2[i] == 4 or c2[i] == 8 or c2[i] == 6 or c2[i] == 3):
                    if c1[i] == 6 or c1[i] == 0:
                        res = 86
                    elif c1[i] == 3 or c1[i] == 7:
                        res = 308
                    elif c1[i] == 2 or c1[i] == 8:
                        res = 456
                    else:
                        res = 504
                else:
                    res = 617
            else:
                if f1[i] < 0.003:
                    res = 57
                elif f1[i] < 0.2518:
                    res = 820
                elif f1[i] < 0.7659:
                    res = 487
                else:
                    res = 100
        if res == -1:
            if (f14[i] == f14[i] and f2[i] > 0.0279 and f10[i] != f10[i]):
                if f1[i] < 0.092:
                    if c1[i] == 8 or c1[i] == 5:
                        res = 179
                    elif c1[i] == 6 or c1[i] == 0:
                        res = 640
                    elif c1[i] == 7 or c1[i] == 3:
                        res = 322
                    else:
                        res = 606
                elif f1[i] < 0.8127:
                    res = 392
                else:
                    if c4[i] == 6 or c4[i] == 1:
                        res = 583
                    elif c4[i] == 3 or c4[i] == 7:
                        res = 206
                    else:
                        res = 281
            else:
                res = 452
        if res == -1:
            if not ((c5[i] == 3 or c5[i] == 7 or c5[i] == 8) and (c0[i] == 8 or c0[i] == 4 or c0[i] == 1 or c0[i] == 0) and not b1[i]):
                if f14[i] > 0.4299:
                    if c0[i] == 0 or c0[i] == 7:
                        res = 772
                    elif c0[i] == 5 or c0[i] == 2:
                        res = 254
                    else:
                        res = 795
                else:
                    if f7[i] < 0.1629:
                        res = 282
                    elif f7[i] < 0.5203:
                        res = 756
                    elif f7[i] < 0.6605:
                        res = 277
                    elif f7[i] < 0.9951:
                        res = 201
                    else:
                        res = 128
            else:
                if f11[i] < 0.2136:
                    res = 126
                elif f11[i] < 0.4242:
                    res = 261
                elif f11[i] < 0.5683:
                    res = 851
                elif f11[i] < 0.6816:
                    res = 286
                else:
                    res = 40
        if res == -1:
            if (c4[i] != 5 and (c4[i] == 3 or c4[i] == 8) and f4[i] > 0.7818):
                if b2[i]:
                    res = 756
                else:
                    if f6[i] == f6[i]:
                        res = 211
                    else:
                        res = 599
            else:
                if c5[i] == 7 or c5[i] == 8:
                    res = 568
                elif c5[i] == 9 or c5[i] == 5:
                    res = 794
                elif c5[i] == 4 or c5[i] == 2:
                    res = 353
                else:
                    res = 834
        if res == -1:
            if (f2[i] >= 0.7005 and c2[i] != 0 and (f7[i] >= 0.1041 and f7[i] <= 0.3245)):
                if not b2[i]:
                    res = 693
                else:
                    if f15[i] < 0.0246:
                        res = 586
                    elif f15[i] < 0.489:
                        res = 610
                    elif f15[i] < 0.6679:
                        res = 730
                    elif f15[i] < 0.7215:
                        res = 574
                    else:
                        res = 54
            else:
                res = 767
        if res == -1:
            if not ((c4[i] == 5 or c4[i] == 2 or c4[i] == 6) or (c1[i] == 9 or c1[i] == 0 or c1[i] == 5)):
                res = 343
            else:
                res = 61
        if res == -1:
            if (c5[i] != 1 or f14[i] > 0.6461 or (c3[i] == 6 or c3[i] == 1 or c3[i] == 5) or (c3[i] == 1 or c3[i] == 6 or c3[i] == 2 or c3[i] == 7)):
                if f6[i] < 0.0182:
                    if c0[i] == 2 or c0[i] == 3:
                        res = 869
                    elif c0[i] == 4 or c0[i] == 7:
                        res = 525
                    elif c0[i] == 8 or c0[i] == 5:
                        res = 898
                    else:
                        res = 52
                elif f6[i] < 0.0625:
                    if c0[i] == 4 or c0[i] == 9:
                        res = 148
                    elif c0[i] == 0 or c0[i] == 2:
                        res = 610
                    elif c0[i] == 7 or c0[i] == 3:
                        res = 845
                    else:
                        res = 52
                elif f6[i] < 0.2305:
                    if (c1[i] == 1 or c1[i] == 7):
                        res = 670
                    else:
                        res = 701
                elif f6[i] < 0.9064:
                    if f1[i] != f1[i]:
                        res = 818
                    else:
                        res = 258
                else:
                    if f7[i] == f7[i]:
                        res = 598
                    else:
                        res = 631
            else:
                res = 452
        if res == -1:
            if (c4[i] != 7 and f8[i] == f8[i]):
                res = 225
            else:
                res = 180
        out[i] = res
