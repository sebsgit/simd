#include <QDebug>
#include <QTemporaryFile>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "simd/simdf4.hpp"

#ifdef Q_OS_ANDROID
static void runTests(int argc, char** argv)
{
    QTemporaryFile tmp;
    tmp.open();
    auto tmpName = tmp.fileName().toUtf8();
    tmp.close();

    auto x = freopen(tmpName.data(), "a+", stdout);

    Catch::Session().run(argc, argv);

    QFile f(tmpName);
    f.open(QIODevice::ReadOnly);

    for (auto& x : f.readAll().split('\n'))
        qDebug() << x;

    fclose(x);
}
#else
static void runTests(int argc, char** argv)
{
    Catch::Session().run(argc, argv);
}
#endif

int main(int argc, char* argv[])
{
    runTests(argc, argv);
    return 0;
}
