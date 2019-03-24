QT -= gui network

CONFIG += console

!android:CONFIG += c++1z
android:CONFIG += c++14

!android:QMAKE_CXXFLAGS += -msse4.1 -mmmx
android:QMAKE_CXXFLAGS += -mfloat-abi=softfp -mfpu=neon

DEFINES += QT_DEPRECATED_WARNINGS
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        main.cpp \
    ../test_simdf4.cpp \
    ../test_simdu16x8.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH+=../ \
	../../

!win32-msvc*:QMAKE_CXXFLAGS += -Wno-error=old-style-cast -Wall

HEADERS += \
    ../catch.hpp \
    ../../simd/simd_base.hpp \
    ../../simd/simdf4.hpp \
    ../../simd/simdu16x8.hpp


android:HEADERS += ../../simd/neon/simdf4_neon.hpp \
    ../../simd/neon/simdu16x8_neon.hpp
!android:HEADERS += ../../simd/x86/simdf4_sse.hpp \
    ../../simd/x86/simdu16x8_sse.hpp
