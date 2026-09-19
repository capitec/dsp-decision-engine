def kern(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, c0, c1, c2, c3, c4, c5, b0, b1, b2, out):
    n = out.shape[0]
    for i in range(n):
        res = -1
        if res == -1:
            if (f12[i] > 0.0821 and (c4[i] == 6 or c4[i] == 4 or c4[i] == 3) and not b1[i] and (c1[i] == 1 or c1[i] == 5 or c1[i] == 0)):
                if c2[i] == 8 or c2[i] == 1:
                    res = 670
                elif c2[i] == 6 or c2[i] == 5:
                    res = 134
                elif c2[i] == 0 or c2[i] == 2:
                    res = 54
                elif c2[i] == 4 or c2[i] == 9:
                    res = 677
                else:
                    if not b0[i]:
                        res = 349
                    else:
                        res = 1
            else:
                res = 387
        if res == -1:
            if (c3[i] != 1 or (f9[i] >= 0.1839 and f9[i] <= 0.5018) or f5[i] < 0.0395):
                if c2[i] == 6 or c2[i] == 4:
                    res = 636
                elif c2[i] == 3 or c2[i] == 2:
                    if f9[i] != f9[i]:
                        res = 394
                    else:
                        res = 273
                elif c2[i] == 0 or c2[i] == 5:
                    res = 212
                else:
                    if c3[i] == 6 or c3[i] == 0:
                        res = 16
                    elif c3[i] == 3 or c3[i] == 8:
                        res = 475
                    elif c3[i] == 1 or c3[i] == 5:
                        res = 879
                    elif c3[i] == 2 or c3[i] == 7:
                        res = 186
                    else:
                        res = 542
            else:
                if c4[i] == 8 or c4[i] == 3:
                    res = 479
                elif c4[i] == 1 or c4[i] == 9:
                    res = 141
                elif c4[i] == 7 or c4[i] == 6:
                    res = 801
                elif c4[i] == 5 or c4[i] == 2:
                    res = 790
                else:
                    res = 171
        if res == -1:
            if (c4[i] != 8 and (f1[i] >= 0.3165 and f1[i] <= 0.4423) and c1[i] != 6 and f6[i] != f6[i]):
                if c5[i] == 0 or c5[i] == 8:
                    res = 618
                elif c5[i] == 1 or c5[i] == 9:
                    res = 316
                elif c5[i] == 4 or c5[i] == 3:
                    if c1[i] == 6 or c1[i] == 0:
                        res = 252
                    elif c1[i] == 3 or c1[i] == 4:
                        res = 451
                    elif c1[i] == 5 or c1[i] == 9:
                        res = 864
                    elif c1[i] == 1 or c1[i] == 7:
                        res = 239
                    else:
                        res = 646
                elif c5[i] == 7 or c5[i] == 6:
                    if f12[i] < 0.2299:
                        res = 56
                    elif f12[i] < 0.4655:
                        res = 581
                    elif f12[i] < 0.7872:
                        res = 635
                    elif f12[i] < 0.9766:
                        res = 452
                    else:
                        res = 447
                else:
                    if (f1[i] >= 0.2806 and f1[i] <= 0.7588):
                        res = 359
                    else:
                        res = 624
            else:
                res = 307
        if res == -1:
            if not (c2[i] != 6 and f4[i] < 0.4333):
                if (c0[i] == 8 or c0[i] == 7 or c0[i] == 9):
                    if f1[i] < 0.4646:
                        res = 264
                    elif f1[i] < 0.5006:
                        res = 105
                    elif f1[i] < 0.8668:
                        res = 557
                    else:
                        res = 306
                else:
                    if c3[i] == 9 or c3[i] == 4:
                        res = 346
                    elif c3[i] == 8 or c3[i] == 6:
                        res = 9
                    elif c3[i] == 1 or c3[i] == 5:
                        res = 79
                    else:
                        res = 721
            else:
                if b2[i]:
                    res = 447
                else:
                    res = 648
        if res == -1:
            if (f11[i] < 0.8399 and b0[i] and (f0[i] >= 0.2433 and f0[i] <= 0.4616)):
                res = 205
            else:
                if c4[i] != 9:
                    res = 793
                else:
                    res = 696
        if res == -1:
            if (c0[i] != 0 or (f11[i] >= 0.2918 and f11[i] <= 0.5918) or c4[i] == 7 or (c5[i] == 8 or c5[i] == 7 or c5[i] == 0)):
                res = 690
            else:
                if f5[i] == f5[i]:
                    res = 379
                else:
                    res = 87
        if res == -1:
            if (f11[i] > 0.1065 or c3[i] == 7 or (c0[i] == 0 or c0[i] == 8 or c0[i] == 4 or c0[i] == 6)):
                if not b2[i]:
                    res = 380
                else:
                    res = 875
            else:
                if c0[i] == 5 or c0[i] == 7:
                    res = 664
                elif c0[i] == 4 or c0[i] == 0:
                    res = 533
                elif c0[i] == 6 or c0[i] == 2:
                    res = 437
                else:
                    res = 776
        if res == -1:
            if (f9[i] < 0.4420 and c5[i] == 1):
                res = 794
            else:
                res = 447
        if res == -1:
            if ((c5[i] == 3 or c5[i] == 6 or c5[i] == 5 or c5[i] == 7) or f0[i] == f0[i] or c4[i] != 0 or (f8[i] >= 0.1411 and f8[i] <= 0.2960)):
                if (f5[i] >= 0.3059 and f5[i] <= 0.5168):
                    res = 394
                else:
                    res = 294
            else:
                if c4[i] == 3 or c4[i] == 8:
                    res = 624
                elif c4[i] == 4 or c4[i] == 0:
                    res = 836
                elif c4[i] == 2 or c4[i] == 1:
                    res = 66
                elif c4[i] == 5 or c4[i] == 7:
                    res = 88
                else:
                    res = 158
        if res == -1:
            if (f6[i] <= 0.5775 and (c5[i] == 7 or c5[i] == 2 or c5[i] == 0 or c5[i] == 5) and b0[i]):
                if (f7[i] >= 0.2468 and f7[i] <= 0.3489):
                    if f4[i] >= 0.3412:
                        res = 282
                    else:
                        res = 653
                else:
                    res = 705
            else:
                if c2[i] != 4:
                    res = 429
                else:
                    res = 119
        if res == -1:
            if (c2[i] != 0 and (f15[i] >= 0.3025 and f15[i] <= 0.7640) and (c3[i] == 6 or c3[i] == 5 or c3[i] == 1 or c3[i] == 0)):
                if f0[i] >= 0.1081:
                    if c2[i] == 2 or c2[i] == 5:
                        res = 291
                    elif c2[i] == 0 or c2[i] == 4:
                        res = 323
                    elif c2[i] == 8 or c2[i] == 1:
                        res = 228
                    else:
                        res = 851
                else:
                    res = 685
            else:
                if c0[i] == 5 or c0[i] == 4:
                    res = 687
                elif c0[i] == 1 or c0[i] == 7:
                    res = 55
                elif c0[i] == 8 or c0[i] == 0:
                    res = 571
                elif c0[i] == 9 or c0[i] == 3:
                    res = 579
                else:
                    res = 610
        if res == -1:
            if (f4[i] >= 0.0871 or (c1[i] == 6 or c1[i] == 8) or f0[i] < 0.7459 or f2[i] >= 0.6376):
                if c1[i] != 8:
                    if f0[i] < 0.0575:
                        res = 851
                    elif f0[i] < 0.2544:
                        res = 499
                    else:
                        res = 851
                else:
                    res = 543
            else:
                res = 734
        if res == -1:
            if ((c0[i] == 2 or c0[i] == 4) and c0[i] == 2):
                if c3[i] == 0 or c3[i] == 9:
                    if c5[i] == 5 or c5[i] == 4:
                        res = 675
                    elif c5[i] == 6 or c5[i] == 1:
                        res = 57
                    else:
                        res = 440
                elif c3[i] == 3 or c3[i] == 4:
                    if f2[i] < 0.3518:
                        res = 152
                    elif f2[i] < 0.655:
                        res = 317
                    elif f2[i] < 0.6644:
                        res = 373
                    else:
                        res = 146
                else:
                    res = 352
            else:
                if f12[i] < 0.4193:
                    res = 862
                elif f12[i] < 0.454:
                    res = 263
                elif f12[i] < 0.5086:
                    res = 558
                elif f12[i] < 0.7663:
                    res = 808
                else:
                    res = 364
        if res == -1:
            if not (f11[i] < 0.8967 and c1[i] == 2 and (f3[i] >= 0.1738 and f3[i] <= 0.3786)):
                if f8[i] == f8[i]:
                    res = 587
                else:
                    if f5[i] < 0.6497:
                        res = 656
                    else:
                        res = 552
            else:
                res = 397
        if res == -1:
            if ((f2[i] >= 0.0628 and f2[i] <= 0.3683) and (c4[i] == 6 or c4[i] == 9)):
                if c2[i] != 0:
                    if f4[i] < 0.0568:
                        res = 212
                    elif f4[i] < 0.0986:
                        res = 490
                    elif f4[i] < 0.3414:
                        res = 349
                    elif f4[i] < 0.5276:
                        res = 884
                    else:
                        res = 422
                else:
                    if f15[i] < 0.086:
                        res = 689
                    elif f15[i] < 0.5007:
                        res = 551
                    elif f15[i] < 0.667:
                        res = 709
                    elif f15[i] < 0.8668:
                        res = 118
                    else:
                        res = 117
            else:
                if c5[i] == 7 or c5[i] == 6:
                    res = 651
                elif c5[i] == 0 or c5[i] == 4:
                    res = 857
                elif c5[i] == 9 or c5[i] == 3:
                    res = 418
                else:
                    res = 543
        if res == -1:
            if not (f15[i] == f15[i] and c2[i] == 7 and b0[i] and b2[i]):
                if c0[i] == 1 or c0[i] == 2:
                    if c5[i] == 9 or c5[i] == 7:
                        res = 817
                    elif c5[i] == 6 or c5[i] == 0:
                        res = 18
                    elif c5[i] == 4 or c5[i] == 1:
                        res = 740
                    else:
                        res = 501
                elif c0[i] == 4 or c0[i] == 3:
                    if (f4[i] >= 0.3856 and f4[i] <= 0.7289):
                        res = 481
                    else:
                        res = 525
                else:
                    if c1[i] == 7 or c1[i] == 4:
                        res = 479
                    elif c1[i] == 2 or c1[i] == 3:
                        res = 833
                    else:
                        res = 854
            else:
                if c3[i] == 4 or c3[i] == 6:
                    res = 690
                elif c3[i] == 3 or c3[i] == 9:
                    res = 542
                else:
                    res = 760
        if res == -1:
            if (b1[i] or f14[i] <= 0.3748 or (c2[i] == 5 or c2[i] == 8 or c2[i] == 9 or c2[i] == 2)):
                res = 59
            else:
                res = 302
        if res == -1:
            if (f3[i] >= 0.9313 and f11[i] <= 0.4424):
                if not b0[i]:
                    if c2[i] == 6 or c2[i] == 0:
                        res = 414
                    elif c2[i] == 1 or c2[i] == 8:
                        res = 729
                    else:
                        res = 232
                else:
                    if c0[i] == 8 or c0[i] == 5:
                        res = 395
                    elif c0[i] == 3 or c0[i] == 7:
                        res = 701
                    elif c0[i] == 6 or c0[i] == 0:
                        res = 635
                    else:
                        res = 217
            else:
                if f3[i] < 0.3627:
                    res = 667
                else:
                    res = 395
        if res == -1:
            if (f1[i] >= 0.9500 and c5[i] == 8):
                if c3[i] == 5 or c3[i] == 8:
                    res = 27
                elif c3[i] == 0 or c3[i] == 3:
                    res = 869
                else:
                    res = 160
            else:
                if f6[i] < 0.0726:
                    res = 795
                elif f6[i] < 0.3285:
                    res = 714
                elif f6[i] < 0.5793:
                    res = 786
                elif f6[i] < 0.7756:
                    res = 841
                else:
                    res = 85
        if res == -1:
            if (f12[i] < 0.4423 or not b2[i] or f7[i] == f7[i] or f5[i] > 0.1024):
                if c1[i] == 9:
                    if f9[i] < 0.3959:
                        res = 121
                    elif f9[i] < 0.7751:
                        res = 547
                    elif f9[i] < 0.8366:
                        res = 381
                    else:
                        res = 255
                else:
                    res = 636
            else:
                if c0[i] == 9 or c0[i] == 0:
                    res = 865
                elif c0[i] == 5 or c0[i] == 2:
                    res = 823
                elif c0[i] == 4 or c0[i] == 7:
                    res = 685
                elif c0[i] == 1 or c0[i] == 3:
                    res = 572
                else:
                    res = 248
        if res == -1:
            if (b2[i] and f9[i] <= 0.0304):
                if c3[i] == 4 or c3[i] == 1:
                    if c1[i] == 9 or c1[i] == 2:
                        res = 398
                    elif c1[i] == 5 or c1[i] == 6:
                        res = 509
                    elif c1[i] == 3 or c1[i] == 1:
                        res = 295
                    elif c1[i] == 8 or c1[i] == 0:
                        res = 196
                    else:
                        res = 623
                elif c3[i] == 0 or c3[i] == 8:
                    if c1[i] == 1 or c1[i] == 9:
                        res = 254
                    elif c1[i] == 4 or c1[i] == 2:
                        res = 612
                    elif c1[i] == 5 or c1[i] == 3:
                        res = 86
                    elif c1[i] == 7 or c1[i] == 8:
                        res = 740
                    else:
                        res = 709
                elif c3[i] == 5 or c3[i] == 7:
                    if c4[i] == 2 or c4[i] == 7:
                        res = 480
                    elif c4[i] == 4 or c4[i] == 3:
                        res = 196
                    else:
                        res = 440
                elif c3[i] == 9 or c3[i] == 3:
                    res = 553
                else:
                    res = 493
            else:
                res = 228
        if res == -1:
            if not ((f5[i] >= 0.2812 and f5[i] <= 0.5572) and f7[i] <= 0.6477):
                if c4[i] == 4 or c4[i] == 0:
                    if f13[i] < 0.3848:
                        res = 791
                    elif f13[i] < 0.4373:
                        res = 674
                    elif f13[i] < 0.8065:
                        res = 448
                    else:
                        res = 385
                elif c4[i] == 2 or c4[i] == 6:
                    if b0[i]:
                        res = 595
                    else:
                        res = 480
                elif c4[i] == 5 or c4[i] == 7:
                    res = 84
                elif c4[i] == 3 or c4[i] == 8:
                    if f0[i] != f0[i]:
                        res = 290
                    else:
                        res = 842
                else:
                    res = 883
            else:
                res = 778
        if res == -1:
            if not (b1[i] or f2[i] <= 0.2593 or (c3[i] == 7 or c3[i] == 3 or c3[i] == 4 or c3[i] == 0) or (c4[i] == 4 or c4[i] == 1)):
                res = 278
            else:
                res = 112
        if res == -1:
            if ((c5[i] == 5 or c5[i] == 6 or c5[i] == 4) or f7[i] >= 0.2051 or not b2[i]):
                res = 333
            else:
                res = 874
        if res == -1:
            if (b0[i] and (c4[i] == 6 or c4[i] == 5) and (c5[i] == 1 or c5[i] == 0)):
                res = 258
            else:
                if f7[i] == f7[i]:
                    res = 888
                else:
                    res = 620
        if res == -1:
            if (not b1[i] or f12[i] != f12[i]):
                res = 781
            else:
                if f14[i] < 0.1175:
                    res = 596
                elif f14[i] < 0.2718:
                    res = 131
                else:
                    res = 625
        if res == -1:
            if (f11[i] < 0.8625 and c2[i] != 2 and f10[i] == f10[i] and (c1[i] == 1 or c1[i] == 0 or c1[i] == 7 or c1[i] == 5)):
                if (c5[i] == 8 or c5[i] == 6):
                    if f1[i] >= 0.5008:
                        res = 807
                    else:
                        res = 745
                else:
                    if f4[i] <= 0.9385:
                        res = 858
                    else:
                        res = 370
            else:
                if b0[i]:
                    res = 640
                else:
                    res = 309
        if res == -1:
            if (c5[i] == 6 or c2[i] == 0 or f2[i] < 0.1787 or f10[i] >= 0.3970):
                if (c1[i] == 3 or c1[i] == 5 or c1[i] == 8):
                    if f6[i] < 0.2277:
                        res = 576
                    elif f6[i] < 0.2317:
                        res = 588
                    elif f6[i] < 0.4916:
                        res = 383
                    elif f6[i] < 0.993:
                        res = 45
                    else:
                        res = 391
                else:
                    res = 863
            else:
                res = 84
        if res == -1:
            if (b2[i] or b2[i]):
                if c4[i] == 8 or c4[i] == 9:
                    if b2[i]:
                        res = 119
                    else:
                        res = 388
                elif c4[i] == 2 or c4[i] == 6:
                    res = 638
                elif c4[i] == 1 or c4[i] == 4:
                    if (c3[i] == 8 or c3[i] == 5):
                        res = 597
                    else:
                        res = 144
                else:
                    if c4[i] == 8 or c4[i] == 7:
                        res = 765
                    elif c4[i] == 1 or c4[i] == 4:
                        res = 497
                    elif c4[i] == 6 or c4[i] == 3:
                        res = 157
                    elif c4[i] == 5 or c4[i] == 0:
                        res = 322
                    else:
                        res = 592
            else:
                res = 312
        if res == -1:
            if (f3[i] != f3[i] or f0[i] <= 0.4635):
                res = 72
            else:
                if c5[i] == 5 or c5[i] == 3:
                    res = 672
                elif c5[i] == 1 or c5[i] == 0:
                    res = 498
                else:
                    res = 200
        out[i] = res
