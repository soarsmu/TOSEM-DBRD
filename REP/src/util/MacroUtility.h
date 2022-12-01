/*
 * TraceMacro.h
 *
 *  Created on: Nov 30, 2010
 *      Author: Chengnian SUN
 */

#ifndef _TRACE_MACRO_H_
#define _TRACE_MACRO_H_

#include <cstdio>
#include <cstdlib>

#define STRING_FORM(name) #name

#define TC_INFO(statement)
// #define TC_INFO(statement) statement

#define UNREACHABLE(message) do { \
	fprintf(stderr, "ERROR: Unreachable to Line %d, File %s, due to reason: %s\n", __LINE__, __FILE__, (message)); \
	exit(EXIT_FAILURE); \
} while(0)

#define ERROR_HERE(message) do {\
	fprintf(stderr, "ERROR: Error at Line %d, File %s, due to reason: %s\n", __LINE__, __FILE__, (message)); \
	exit(EXIT_FAILURE); \
} while(0)

#define EXIT_ON_ERROR(message) do {\
	fprintf(stderr, "ERROR: %s\n", (message)); \
	exit(EXIT_FAILURE); \
} while(0)

#endif /* _TRACE_MACRO_H_ */
