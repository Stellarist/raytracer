#include <cstdio>

__global__ void hello()
{
    printf(% s, "hello, world");
}

int main(int argc, char const *argv[])
{
    hello<<<1, 1>>>();
    return 0;
}
