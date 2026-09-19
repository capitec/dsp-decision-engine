def kern(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, c0, c1, c2, c3, c4, c5, b0, b1, b2, out):
    n = out.shape[0]
    for i in range(n):
        res = -1
        if res == -1:
            if ((f12[i] >= 0.4365 and f12[i] <= 0.6149) or f13[i] <= 0.4427 or (c0[i] == 5 or c0[i] == 3 or c0[i] == 8 or c0[i] == 7)):
                if c1[i] == 9 or c1[i] == 4:
                    if c3[i] == 3 or c3[i] == 7:
                        res = 12
                    elif c3[i] == 1 or c3[i] == 9:
                        res = 182
                    elif c3[i] == 8 or c3[i] == 5:
                        res = 219
                    else:
                        res = 207
                elif c1[i] == 2 or c1[i] == 0:
                    res = 131
                elif c1[i] == 8 or c1[i] == 3:
                    if c1[i] == 0 or c1[i] == 8:
                        res = 887
                    elif c1[i] == 7 or c1[i] == 9:
                        res = 472
                    elif c1[i] == 1 or c1[i] == 2:
                        res = 785
                    elif c1[i] == 5 or c1[i] == 4:
                        res = 249
                    else:
                        res = 358
                elif c1[i] == 1 or c1[i] == 7:
                    if c3[i] == 0 or c3[i] == 7:
                        res = 354
                    elif c3[i] == 9 or c3[i] == 4:
                        res = 114
                    else:
                        res = 203
                else:
                    if f2[i] <= 0.9571:
                        res = 517
                    else:
                        res = 400
            else:
                if c0[i] != 4:
                    res = 756
                else:
                    res = 173
        if res == -1:
            if (f15[i] == f15[i] or b1[i] or f4[i] == f4[i]):
                if (c1[i] == 0 or c1[i] == 9 or c1[i] == 2):
                    res = 666
                else:
                    if b2[i]:
                        res = 140
                    else:
                        res = 28
            else:
                if f5[i] > 0.6371:
                    res = 64
                else:
                    res = 825
        if res == -1:
            if not (f15[i] < 0.3837 or c1[i] != 8 or f2[i] == f2[i] or f13[i] > 0.4675):
                if not b1[i]:
                    if not b2[i]:
                        res = 240
                    else:
                        res = 871
                else:
                    if (c4[i] == 1 or c4[i] == 3):
                        res = 465
                    else:
                        res = 482
            else:
                if f2[i] > 0.5539:
                    res = 21
                else:
                    res = 44
        if res == -1:
            if (f10[i] <= 0.5061 or b2[i] or (c5[i] == 6 or c5[i] == 7 or c5[i] == 5)):
                res = 103
            else:
                if (c5[i] == 7 or c5[i] == 9 or c5[i] == 6 or c5[i] == 4):
                    res = 569
                else:
                    res = 122
        if res == -1:
            if (f9[i] > 0.6613 or (c0[i] == 4 or c0[i] == 3 or c0[i] == 5 or c0[i] == 9) or (f6[i] >= 0.3091 and f6[i] <= 0.4685)):
                res = 244
            else:
                if c1[i] == 7 or c1[i] == 3:
                    res = 454
                elif c1[i] == 0 or c1[i] == 2:
                    res = 22
                else:
                    res = 719
        if res == -1:
            if (not b1[i] or c1[i] != 2):
                res = 865
            else:
                res = 22
        if res == -1:
            if not (not b2[i] and (f12[i] >= 0.1599 and f12[i] <= 0.4937)):
                if c5[i] != 2:
                    if c3[i] == 0 or c3[i] == 4:
                        res = 463
                    elif c3[i] == 6 or c3[i] == 7:
                        res = 646
                    elif c3[i] == 1 or c3[i] == 9:
                        res = 760
                    elif c3[i] == 8 or c3[i] == 3:
                        res = 243
                    else:
                        res = 293
                else:
                    res = 23
            else:
                if (c5[i] == 3 or c5[i] == 8):
                    res = 20
                else:
                    res = 353
        if res == -1:
            if (f7[i] < 0.3325 or f11[i] > 0.6230 or c3[i] == 9):
                res = 444
            else:
                if c2[i] == 1 or c2[i] == 7:
                    res = 866
                elif c2[i] == 4 or c2[i] == 2:
                    res = 682
                elif c2[i] == 6 or c2[i] == 5:
                    res = 713
                elif c2[i] == 0 or c2[i] == 3:
                    res = 180
                else:
                    res = 720
        if res == -1:
            if ((f0[i] >= 0.2922 and f0[i] <= 0.5862) or f2[i] < 0.4966):
                if c3[i] == 6 or c3[i] == 7:
                    res = 351
                elif c3[i] == 3 or c3[i] == 9:
                    if (c1[i] == 5 or c1[i] == 2 or c1[i] == 3 or c1[i] == 8):
                        res = 656
                    else:
                        res = 897
                elif c3[i] == 0 or c3[i] == 2:
                    if f4[i] <= 0.0144:
                        res = 146
                    else:
                        res = 632
                else:
                    if f15[i] < 0.5481:
                        res = 306
                    elif f15[i] < 0.6193:
                        res = 691
                    elif f15[i] < 0.6634:
                        res = 497
                    else:
                        res = 273
            else:
                res = 340
        if res == -1:
            if (f6[i] != f6[i] or c3[i] != 0 or (c0[i] == 9 or c0[i] == 8 or c0[i] == 6 or c0[i] == 3)):
                if c3[i] == 4:
                    res = 845
                else:
                    if c4[i] == 3 or c4[i] == 8:
                        res = 534
                    elif c4[i] == 7 or c4[i] == 9:
                        res = 743
                    else:
                        res = 727
            else:
                res = 882
        if res == -1:
            if (f2[i] <= 0.0727 or f12[i] >= 0.9515 or f13[i] >= 0.5150):
                res = 895
            else:
                if f0[i] <= 0.5583:
                    res = 422
                else:
                    res = 246
        if res == -1:
            if not (not b2[i] and c0[i] == 8 and c3[i] != 2):
                res = 567
            else:
                if f11[i] > 0.9951:
                    res = 162
                else:
                    res = 381
        if res == -1:
            if not (f3[i] >= 0.2052 and c5[i] != 9 and f12[i] != f12[i] and f3[i] > 0.2648):
                if f12[i] < 0.0375:
                    res = 253
                elif f12[i] < 0.4388:
                    if f0[i] < 0.0292:
                        res = 151
                    elif f0[i] < 0.3895:
                        res = 317
                    elif f0[i] < 0.6083:
                        res = 486
                    else:
                        res = 307
                elif f12[i] < 0.5474:
                    res = 484
                else:
                    if (c0[i] == 7 or c0[i] == 0 or c0[i] == 6 or c0[i] == 8):
                        res = 144
                    else:
                        res = 278
            else:
                res = 394
        if res == -1:
            if (f11[i] != f11[i] and c5[i] == 6 and f10[i] <= 0.6963):
                if f2[i] < 0.049:
                    if c2[i] == 4 or c2[i] == 8:
                        res = 728
                    elif c2[i] == 7 or c2[i] == 2:
                        res = 630
                    else:
                        res = 589
                elif f2[i] < 0.292:
                    if c4[i] == 6:
                        res = 499
                    else:
                        res = 402
                elif f2[i] < 0.4847:
                    res = 762
                elif f2[i] < 0.7809:
                    if c5[i] == 2 or c5[i] == 9:
                        res = 768
                    elif c5[i] == 1 or c5[i] == 5:
                        res = 408
                    elif c5[i] == 3 or c5[i] == 6:
                        res = 851
                    else:
                        res = 736
                else:
                    if f3[i] == f3[i]:
                        res = 613
                    else:
                        res = 861
            else:
                if f13[i] != f13[i]:
                    res = 623
                else:
                    res = 579
        if res == -1:
            if (b2[i] and not b2[i]):
                res = 709
            else:
                if c0[i] == 8 or c0[i] == 1:
                    res = 728
                elif c0[i] == 5 or c0[i] == 4:
                    res = 232
                elif c0[i] == 2 or c0[i] == 7:
                    res = 275
                else:
                    res = 618
        if res == -1:
            if (f11[i] > 0.1680 and (f5[i] >= 0.2102 and f5[i] <= 0.5713)):
                res = 419
            else:
                res = 790
        if res == -1:
            if (f3[i] < 0.3588 or b1[i]):
                if not b0[i]:
                    res = 826
                else:
                    if f9[i] == f9[i]:
                        res = 624
                    else:
                        res = 590
            else:
                res = 778
        if res == -1:
            if (f10[i] <= 0.5138 or b0[i] or f13[i] > 0.0164):
                if c3[i] == 1 or c3[i] == 7:
                    if f10[i] >= 0.8442:
                        res = 635
                    else:
                        res = 746
                elif c3[i] == 5 or c3[i] == 2:
                    res = 791
                elif c3[i] == 4 or c3[i] == 3:
                    if c5[i] == 8 or c5[i] == 2:
                        res = 80
                    elif c5[i] == 9 or c5[i] == 0:
                        res = 642
                    elif c5[i] == 4 or c5[i] == 1:
                        res = 687
                    else:
                        res = 106
                elif c3[i] == 6 or c3[i] == 9:
                    if f15[i] != f15[i]:
                        res = 277
                    else:
                        res = 816
                else:
                    res = 722
            else:
                res = 445
        if res == -1:
            if (f7[i] < 0.7735 or f7[i] > 0.7859):
                if c0[i] == 2 or c0[i] == 9:
                    res = 49
                elif c0[i] == 0 or c0[i] == 6:
                    res = 379
                else:
                    if f12[i] < 0.1483:
                        res = 568
                    elif f12[i] < 0.2472:
                        res = 800
                    elif f12[i] < 0.3818:
                        res = 534
                    elif f12[i] < 0.7674:
                        res = 268
                    else:
                        res = 62
            else:
                if c0[i] == 5 or c0[i] == 6:
                    res = 828
                elif c0[i] == 1 or c0[i] == 7:
                    res = 498
                elif c0[i] == 3 or c0[i] == 8:
                    res = 401
                else:
                    res = 57
        if res == -1:
            if (f0[i] == f0[i] or f7[i] < 0.6758 or f3[i] > 0.4277):
                if f8[i] >= 0.4204:
                    if (f11[i] >= 0.2960 and f11[i] <= 0.7749):
                        res = 426
                    else:
                        res = 877
                else:
                    if c4[i] == 6 or c4[i] == 3:
                        res = 835
                    elif c4[i] == 0 or c4[i] == 8:
                        res = 352
                    elif c4[i] == 7 or c4[i] == 9:
                        res = 825
                    else:
                        res = 338
            else:
                if (c1[i] == 9 or c1[i] == 4 or c1[i] == 1 or c1[i] == 3):
                    res = 286
                else:
                    res = 855
        if res == -1:
            if (f6[i] == f6[i] or f9[i] != f9[i]):
                res = 701
            else:
                if (c1[i] == 9 or c1[i] == 8 or c1[i] == 2 or c1[i] == 6):
                    res = 239
                else:
                    res = 296
        if res == -1:
            if ((f0[i] >= 0.4883 and f0[i] <= 0.7650) and (c1[i] == 4 or c1[i] == 0 or c1[i] == 5)):
                if f10[i] <= 0.9551:
                    if (c2[i] == 5 or c2[i] == 1):
                        res = 203
                    else:
                        res = 736
                else:
                    res = 286
            else:
                if (c3[i] == 3 or c3[i] == 6 or c3[i] == 5 or c3[i] == 2):
                    res = 622
                else:
                    res = 583
        if res == -1:
            if (c5[i] == 5 or b2[i] or f9[i] <= 0.2058):
                if c0[i] == 1 or c0[i] == 2:
                    res = 160
                elif c0[i] == 8 or c0[i] == 7:
                    if f12[i] >= 0.3467:
                        res = 814
                    else:
                        res = 728
                elif c0[i] == 5 or c0[i] == 9:
                    if f0[i] < 0.3581:
                        res = 585
                    elif f0[i] < 0.6095:
                        res = 601
                    elif f0[i] < 0.869:
                        res = 176
                    elif f0[i] < 0.9925:
                        res = 502
                    else:
                        res = 708
                elif c0[i] == 4 or c0[i] == 6:
                    if c2[i] == 3 or c2[i] == 5:
                        res = 136
                    elif c2[i] == 7 or c2[i] == 8:
                        res = 229
                    elif c2[i] == 2 or c2[i] == 9:
                        res = 335
                    elif c2[i] == 1 or c2[i] == 6:
                        res = 363
                    else:
                        res = 366
                else:
                    if f11[i] < 0.1335:
                        res = 568
                    elif f11[i] < 0.3167:
                        res = 777
                    elif f11[i] < 0.6577:
                        res = 871
                    elif f11[i] < 0.7417:
                        res = 855
                    else:
                        res = 822
            else:
                if c3[i] == 6 or c3[i] == 9:
                    res = 384
                elif c3[i] == 5 or c3[i] == 0:
                    res = 697
                else:
                    res = 338
        if res == -1:
            if (f7[i] < 0.8194 and not b0[i] and f5[i] != f5[i]):
                res = 207
            else:
                if c2[i] == 9 or c2[i] == 3:
                    res = 374
                elif c2[i] == 0 or c2[i] == 4:
                    res = 809
                else:
                    res = 267
        if res == -1:
            if (f14[i] > 0.5686 and (c1[i] == 0 or c1[i] == 2)):
                if f13[i] < 0.3451:
                    if f13[i] >= 0.6958:
                        res = 891
                    else:
                        res = 155
                else:
                    res = 107
            else:
                res = 522
        if res == -1:
            if (f13[i] > 0.0941 and c1[i] != 7):
                res = 814
            else:
                res = 331
        if res == -1:
            if (f11[i] >= 0.9657 or f15[i] == f15[i]):
                if f5[i] >= 0.1765:
                    if f14[i] < 0.9799:
                        res = 395
                    else:
                        res = 11
                else:
                    res = 674
            else:
                res = 53
        if res == -1:
            if not (c3[i] != 9 or f9[i] >= 0.5044):
                res = 895
            else:
                if c2[i] == 0 or c2[i] == 6:
                    res = 178
                elif c2[i] == 7 or c2[i] == 8:
                    res = 47
                else:
                    res = 656
        if res == -1:
            if (f9[i] >= 0.3659 and (c4[i] == 9 or c4[i] == 7 or c4[i] == 4 or c4[i] == 5) and f9[i] <= 0.6303):
                if c0[i] != 6:
                    if f0[i] != f0[i]:
                        res = 326
                    else:
                        res = 403
                else:
                    if c2[i] == 6:
                        res = 820
                    else:
                        res = 13
            else:
                res = 598
        if res == -1:
            if (c1[i] == 7 or (c4[i] == 6 or c4[i] == 1)):
                if (c0[i] == 8 or c0[i] == 5):
                    if c5[i] == 4:
                        res = 623
                    else:
                        res = 171
                else:
                    res = 407
            else:
                if (c1[i] == 5 or c1[i] == 1 or c1[i] == 7 or c1[i] == 0):
                    res = 595
                else:
                    res = 109
        out[i] = res
