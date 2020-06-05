#include <iostream>
#include <string>
#include <vector>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include <chrono>

using namespace std::chrono;

template <char not_case_lower_bound, char not_case_upper_bound>
struct LowerUpperImpl
{


public:
    static void array( char * src,  char * src_end, char * dst)
    {
        //32
        const auto flip_case_mask = 'A' ^ 'a';

#ifdef __SSE2__
        const auto bytes_sse = sizeof(__m128i);
        const auto src_end_sse = src_end - (src_end - src) % bytes_sse;

        const auto v_not_case_lower_bound = _mm_set1_epi8(not_case_lower_bound - 1);
        const auto v_not_case_upper_bound = _mm_set1_epi8(not_case_upper_bound + 1);
        const auto v_flip_case_mask = _mm_set1_epi8(flip_case_mask);

        for (; src < src_end_sse; src += bytes_sse, dst += bytes_sse)
        {
            //_mm_loadu_si128表示：Loads 128-bit value；即加载128位值。
            //一次性加载16个连续的8-bit字符
            const auto chars = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));

            //_mm_and_si128(a,b)表示：将a和b进行与运算，即r=a&b
            //_mm_cmpgt_epi8(a,b)表示：分别比较a的每个8bits整数是否大于b的对应位置的8bits整数，若大于，则返回0xff，否则返回0x00。
            //_mm_cmplt_epi8(a,b)表示：分别比较a的每个8bits整数是否小于b的对应位置的8bits整数，若小于，则返回0xff，否则返回0x00。
            //下面的一行代码对这128位的寄存器并行操作了3遍，最后得到一个128位数，对应位置上是0xff的，表示
            //那个8-bit数在 [case_lower_bound, case_upper_bound]范围之内的，其余的被0占据的位置，是不在操作范围内的数。
            const auto is_not_case
                = _mm_and_si128(_mm_cmpgt_epi8(chars, v_not_case_lower_bound), _mm_cmplt_epi8(chars, v_not_case_upper_bound));

            //每个0xff位置与32进行与操作，原来的oxff位置变成32，也就是说，每个在 [case_lower_bound, case_upper_bound]范围区间的数，现在变成了32，其他的位置是0
            const auto xor_mask = _mm_and_si128(v_flip_case_mask, is_not_case);

            //将源chars内容与xor_mask进行异或，符合条件的字节可能从uppercase转为lowercase，也可能从lowercase转为uppercase，不符合区间的仍保留原样。
            const auto cased_chars = _mm_xor_si128(chars, xor_mask);

            //将结果集存到dst中
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), cased_chars);
        }

#endif

#ifndef __SSE2__
        for (; src < src_end; ++src, ++dst)
            if (*src >= not_case_lower_bound && *src <= not_case_upper_bound)
                *dst = *src ^ flip_case_mask;
            else
                *dst = *src;
#endif
    }

};

int main()
{
    char all[128] = {'A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H'};
    char des[129] = {'\0'};
    static LowerUpperImpl<'A','Z'> tmp;

    //获取开始时间
    auto start = system_clock::now();
    for (size_t i = 0; i < 1; ++i){
        tmp.array(&all[0],&all[128],&des[0]);
    }
    std::cout <<"new des is "<<des<<std::endl;
    //获取结束时间
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout <<"take time:"<< double(duration.count()) * microseconds::period::num/microseconds::period::den <<"s"<<std::endl;
}
