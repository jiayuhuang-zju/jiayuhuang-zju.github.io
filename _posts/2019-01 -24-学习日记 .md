# 2019 1 24
```
MIT6.858 Lec1&Lec2
信息系统安全
Freebuf
```
## MIT6.858
- What's security：Achieving some goal in the presence of an adversary .
- High-level plan for thinking security:
	- Police
	- Common goals: confidentiality,intergrity,availability.
	- Thread model
	- Mechanism
	- Resulting goal
### example code
```C
Int read_req(void){
	char buf[128];
	int I;
	gets(buf);
	I = atoi(buf);
	return I;
```
How does the adversary take advantage of this code?

- Supply long input to overwrite data on stack past buffer.
- Interesting bit of data: return address, gets used by 'ret';
- set return address to the buffer itself.

How does the adversary know the address of the buffer?

- What if one machine has twice as much memory?(?)
- Luckily for adversary, virtual memory makes things more deterministic.(?)
- For a given OS and program, address will often the same.(ALSR? How to adverse?)

Any memory errors can translate into a vulnerability

- Using memory after it has been deallocated(use-after-free)
	- If writing, overwrite new data structure.
	- If reading, might call a corrupted function pointer.
	- [uaf利用过程](https://www.anquanke.com/post/id/85281)：
		- 申请一段空间，后将其释放，释放后并不将其指针置为空，意味着这个p1指针仍指向这个空间。
		- 再次申请空间，由于malloc分配的过程使得p2指针指向的空间为刚刚释放的p1指针的空间**(dllmalloc)**，构造恶意的数据将这段内存空间布置好。
		- 利用p1，一般多有一个函数指针(?),由于之前使用的p2覆盖了p1的数据，因此这时的数据我们可以控制，即我们利用p1劫持了函数流
	- [stackexchange](https://security.stackexchange.com/questions/20371/from-a-technical-standpoint-how-does-the-zero-day-internet-explorer-vulnerabili)
- Free the same memory twice(double free)
	- free函数在释放堆块时，会通过隐式链表判断相邻前、后堆块是否为空闲堆块；如果堆块为空闲则进行合并，然后利用Unlink机制将该空闲堆块从Unsorted bin中取下，如果用户精心构造的假堆块被Unlink，很容易导致一次固定地址写，然后转换位任意地址写，从而控制程序的执行。[参考链接](http://d0m021ng.github.io/2017/02/24/PWN/Linux堆漏洞之Double-free/)
	- CTF Link:
		- [Double free 浅析](https://wooyun.js.org/drops/Double%20Free%E6%B5%85%E6%9E%90.html)
		- [将一个指针 free 两次之后会发生什么？](https://zhuanlan.zhihu.com/p/30513886)
- Decrementing the stack ptr past the end of stack, into some other memory.
	- 破坏堆中内存分配信息数据，特别是动态分配的内存块的内存信息数据；破坏程序自己的其他对象的空间，破坏空闲指针块
		
		```*** glibc detected *** free(): invalid pointer:
        *** glibc detected *** malloc(): memory corruption:
        *** glibc detected *** double free or corruption (out): 0x00000000005c18a0 ***
        *** glibc detected *** corrupted double-linked list: 0x00000000005ab150 **‌
	    ```
	- 内存越界定位方法：查看越界的那段内存，之后查看内存实际使用情况，看看是否有异常！大多是数组越界或者字符串的拷贝问题。
- [A one-byte stray write can lead to compromise.](https://www.openwall.com/lists/oss-security/2014/08/26/2)
	- [CTF中off-by-one](https://ctf-wiki.github.io/ctf-wiki/pwn/linux/glibc-heap/off_by_one/)
- [CTF|PWN堆溢出总结](https://www.freebuf.com/articles/system/171261.html)
- [CTF-Wiki](https://ctf-wiki.github.io/ctf-wiki/)


How to avoid mechanism problems?

- Reduce the amount of security-critical code.
- Avoid bugs in security-critical code.
- Examples of common mechanisms:
	- OS-level access comtrol
	- Network firewalls
	- Cryptography, cryptographic protocols.