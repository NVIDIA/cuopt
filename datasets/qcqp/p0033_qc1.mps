NAME          p0033_qc1
ROWS
 N  R100
 L  R118
 L  R119
 L  R120
 L  R121
 L  R122
 L  R123
 L  R124
 L  R125
 L  R126
 L  R127
 L  R128
 L  ZBESTROW
 L  QC1
 L  QC2
 L  QC3
 L  QC4
COLUMNS
    C157      R100                171
    C157      R122               -300
    C157      R123               -300
    C158      R100                171
    C158      R126               -300
    C158      R127               -300
    C159      R100                171
    C159      R119                300
    C159      R120               -300
    C159      R121               -300
    C159      QC1                   1
    C160      R100                171
    C160      R119                300
    C160      R120               -300
    C160      R121               -300
    C161      R100                163
    C161      R119                285
    C161      R120               -285
    C161      R124               -285
    C161      R125               -285
    C162      R100                162
    C162      R119                285
    C162      R120               -285
    C162      R122               -285
    C162      R123               -285
    C163      R100                163
    C163      R128               -285
    C164      R100                 69
    C164      R119                265
    C164      R120               -265
    C164      R124               -265
    C164      R125               -265
    C165      R100                 69
    C165      R119                265
    C165      R120               -265
    C165      R122               -265
    C165      R123               -265
    C166      R100                183
    C166      R118               -230
    C167      R100                183
    C167      R124               -230
    C167      R125               -230
    C168      R100                183
    C168      R119                230
    C168      R120               -230
    C168      R125               -230
    C169      R100                183
    C169      R119                230
    C169      R120               -230
    C169      R123               -230
    C170      R100                 49
    C170      R119                190
    C170      R120               -190
    C170      R122               -190
    C170      R123               -190
    C171      R100                183
    C172      R100                258
    C172      R118               -200
    C173      R100                517
    C173      R118               -400
    C174      R100                250
    C174      R126               -200
    C174      R127               -200
    C175      R100                500
    C175      R126               -400
    C175      R127               -400
    C176      R100                250
    C176      R127               -200
    C177      R100                500
    C177      R127               -400
    C178      R100                159
    C178      R119                200
    C178      R120               -200
    C178      R124               -200
    C178      R125               -200
    C179      R100                318
    C179      R119                400
    C179      R120               -400
    C179      R124               -400
    C179      R125               -400
    C180      R100                159
    C180      R119                200
    C180      R120               -200
    C180      R125               -200
    C181      R100                318
    C181      R119                400
    C181      R120               -400
    C181      R125               -400
    C182      R100                159
    C182      R119                200
    C182      R120               -200
    C182      R122               -200
    C182      R123               -200
    C183      R100                318
    C183      R119                400
    C183      R120               -400
    C183      R122               -400
    C183      R123               -400
    C184      R100                159
    C184      R119                200
    C184      R120               -200
    C184      R123               -200
    C185      R100                318
    C185      R119                400
    C185      R120               -400
    C185      R123               -400
    C186      R100                114
    C186      R119                200
    C186      R120               -200
    C186      R121               -200
    C187      R100                228
    C187      R119                400
    C187      R120               -400
    C187      R121               -400
    C188      R100                159
    C188      R128               -200
    C189      R100                318
    C189      R128               -400
RHS
    rhs       R118                  -5
    rhs       R119                2700
    rhs       R120               -2600
    rhs       R121                -100
    rhs       R122                -900
    rhs       R123               -1656
    rhs       R124                -335
    rhs       R125               -1026
    rhs       R126                  -5
    rhs       R127                -500
    rhs       R128                -270
    rhs       QC1                    1
    rhs       QC2                    2
    rhs       QC3                    1
    rhs       QC4                    1
BOUNDS
 UP bnd       C157                   1
 UP bnd       C158                   1
 UP bnd       C159                   1
 UP bnd       C160                   1
 UP bnd       C161                   1
 UP bnd       C162                   1
 UP bnd       C163                   1
 UP bnd       C164                   1
 UP bnd       C165                   1
 UP bnd       C166                   1
 UP bnd       C167                   1
 UP bnd       C168                   1
 UP bnd       C169                   1
 UP bnd       C170                   1
 UP bnd       C171                   1
 UP bnd       C172                   1
 UP bnd       C173                   1
 UP bnd       C174                   1
 UP bnd       C175                   1
 UP bnd       C176                   1
 UP bnd       C177                   1
 UP bnd       C178                   1
 UP bnd       C179                   1
 UP bnd       C180                   1
 UP bnd       C181                   1
 UP bnd       C182                   1
 UP bnd       C183                   1
 UP bnd       C184                   1
 UP bnd       C185                   1
 UP bnd       C186                   1
 UP bnd       C187                   1
 UP bnd       C188                   1
 UP bnd       C189                   1
QMATRIX
    C158      C158                   1
    C158      C189                 0.5
    C189      C158                 0.5
    C189      C189                   1
QCMATRIX   QC1
    C157      C157                   1
    C157      C158                 0.5
    C158      C157                 0.5
    C158      C158                   1
    C159      C159                   1
    C160      C160                   1
QCMATRIX   QC2
    C161      C161                   2
    C162      C162                   2
    C163      C163                   1
QCMATRIX   QC3
    C164      C164                   1
    C165      C165                   1
QCMATRIX   QC4
    C166      C166                   1
    C167      C167                   1
    C168      C168                   1
    C169      C169                   1
    C171      C171                   1
ENDATA